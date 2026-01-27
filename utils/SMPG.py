import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla


class Variable:
    def __init__(self, main):
        self.main = main
        self.Ax = None
        self.trace = None


def smpg(A, gamma, opts=None):
    if opts is None:
        opts = {}

    # Specify the initial point
    if 'x0' in opts and opts['x0'] is not None:
        # Check if x0 is a Variable object or just a matrix
        if hasattr(opts['x0'], 'main'):
            x0 = opts['x0']
        else:
            x0 = Variable(opts['x0'])
        n, p = x0.main.shape
    elif 'dim' in opts:
        n = opts['dim'][0]
        p = opts['dim'][1]
        if n < p:
            raise ValueError(
                'Column size should be no less than the size of row. Please check the dimensions of the opts.dim.'
            )

        # Random orthogonal initialization
        q, _ = la.qr(np.random.randn(n, p), mode='economic')
        x0 = Variable(q)
    else:
        raise ValueError(
            'No initial point, and the size of initial point is not specified'
        )

    # Options
    tol = opts.get('tol', 1e-6)
    maxiter = opts.get('maxiter', 1000)
    L = opts.get('L', 1.0)
    rho = opts.get('rho', 1.0)
    beta = opts.get('beta', 0.5)
    mu = opts.get('mu', 0.01)
    theta = opts.get('theta', 0.5)
    mode = opts.get('mode', 'l1')

    # Parameters for line search
    upsilon = 0.0001

    # Functions for the optimization problem
    # gfhandle = lambda x, mu: grad(x, A, rho, mu) # Defined later
    fcalJ = calJ

    norm_As_sq = np.linalg.norm(A, 'fro') ** 2

    if mode == 'l1':
        fhandle = lambda x, mu: func_l1(x, A, gamma, rho, mu, norm_As_sq)
        fprox = prox_l1
    elif mode == 'l21':
        fhandle = lambda x, mu: func_l21(x, A, gamma, rho, mu, norm_As_sq)
        fprox = prox_l21
    else:
        raise ValueError(f"Unknown mode: {mode}")

    gfhandle = lambda x, mu: grad(x, A, rho, mu)

    # Functions for the manifold
    fcalA = calA
    fcalAstar = calAstar

    xinitial = x0
    xopt_var, fs, Ds = solver(
        fhandle,
        gfhandle,
        fcalA,
        fcalAstar,
        fprox,
        fcalJ,
        xinitial,
        L,
        tol,
        upsilon,
        beta,
        mu,
        theta,
        maxiter,
        gamma,
    )

    x_main = xopt_var.main

    sparsity = 0
    if mode == 'l1':
        # Thresholding similar to MATLAB code
        x_main[np.abs(x_main) < 1e-5] = 0
        sparsity = np.sum(np.abs(x_main) < 1e-5) / (n * p)
    elif mode == 'l21':
        norm_row = np.linalg.norm(x_main, axis=1)
        inact_set = norm_row < 1e-3
        x_main[inact_set, :] = 0
        sparsity = np.count_nonzero(inact_set) / n

    output = {
        'xopt': x_main,
        'iter': len(fs),
        'fopt': fs[-1] if len(fs) > 0 else 0,
        'spar': sparsity,
        'nD': Ds[-1] if len(Ds) > 0 else 0,
        'fvals': fs,
    }

    return output


def func_l1(x, A, gamma, rho, mu, norm_As=None):
    x.Ax = A @ x.main
    norm_Axs = np.linalg.norm(x.Ax, 'fro') ** 2
    if norm_As is None:
        norm_As = np.linalg.norm(A, 'fro') ** 2
    x.trace = norm_As - norm_Axs

    val = x.trace + gamma * np.sum(np.abs(x.main))

    if x.trace >= mu / 2:
        val = val + 2 * rho * (x.trace) ** (0.5)
    else:
        val = val + 2 * rho * ((x.trace) ** 2 / mu + mu / 4) ** (0.5)

    return val, x


def func_l21(x, A, gamma, rho, mu, norm_As=None):
    x.Ax = A @ x.main
    norm_Axs = np.linalg.norm(x.Ax, 'fro') ** 2
    if norm_As is None:
        norm_As = np.linalg.norm(A, 'fro') ** 2
    x.trace = norm_As - norm_Axs

    # vecnorm(x.main, 2, 2) in MATLAB is norm along rows
    val = (
        x.trace
        + 2 * rho * (x.trace + mu) ** (0.5)
        + gamma * np.sum(np.linalg.norm(x.main, axis=1))
    )
    return val, x


def grad(x, A, rho, mu):
    # gfx = -2 * (A' * x.Ax)
    gfx = -2 * (A.T @ x.Ax)

    if x.trace >= mu / 2:
        # gfx = gfx + rho * (x.trace)^(-1/2) * gfx
        # Since gfx is the gradient of x.trace part? Wait.
        # Original: gfx = - 2 * (A' * x.Ax); this is grad of x.trace.
        # Let g_trace = -2 * A' * A * x.
        # If output includes h(trace), then grad is h'(trace) * g_trace.
        # h(t) = 2*rho*t^(1/2). h'(t) = rho * t^(-1/2).
        # So gfx_total = g_trace + h'(trace) * g_trace = (1 + h'(trace)) * g_trace?
        # MATLAB: gfx = gfx + rho * (x.trace)^(-1/2) * gfx;
        # Yes, equivalent to (1 + rho*trace^(-1/2)) * gfx
        gfx = gfx + rho * (x.trace) ** (-0.5) * gfx
    else:
        # h(t) = 2*rho * sqrt(t^2/mu + mu/4).
        # h'(t) = 2*rho * 0.5 * (t^2/mu + mu/4)^(-1/2) * (2t/mu)
        #       = 2*rho * (t^2/mu + mu/4)^(-1/2) * t/mu
        # MATLAB: gfx = gfx + rho * ((x.trace)^2 / mu + mu / 4)^(-1/2) * 2 * x.trace / mu * gfx
        # Matches.
        gfx = (
            gfx
            + rho * ((x.trace) ** 2 / mu + mu / 4) ** (-0.5) * (2 * x.trace / mu) * gfx
        )

    # Project to tangent space gradient?
    # tmp = gfx' * x.main
    # output = gfx - x.main * ((tmp + tmp') / 2)
    # This is calculating the Riemannian gradient from Euclidean gradient for Stiefel Manifold.
    # grad_R f(X) = grad f(X) - X sym(X^T grad f(X))
    tmp = gfx.T @ x.main
    output = gfx - x.main @ ((tmp + tmp.T) / 2)
    return output


def prox_l1(X, t, gamma):
    # output = min(0, X + t * gamma) + max(0, X - t * gamma)
    # This is soft thresholding
    return np.maximum(0, X - t * gamma) - np.maximum(0, -X - t * gamma)
    # Equivalently: np.sign(X) * np.maximum(np.abs(X) - t*gamma, 0)
    # But let's stick to the formula in MATLAB
    # min(0, Y) is negative part.
    # if X > t*gamma: min(0, >0) + max(0, >0) -> 0 + X-t*gamma
    # if -t*gamma < X < t*gamma: min(0, >0) + max(0, <0) -> 0 + 0 -> 0
    # if X < -t*gamma: min(0, <0) + max(0, <0) -> X+t*gamma + 0
    # It seems correct.


def prox_l21(X, t, lam):
    # proximal mapping of L_21 norm (row-wise)
    r_dim, _ = X.shape
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    a = nr - t * lam

    # Act_set = double(a > 0)
    # In MATLAB, if r < 15, double(a>0), else (a>0). Both are 0/1 (or boolean that acts as 0/1).
    Act_set = (a > 0).astype(float)  # shape (r_dim, 1)

    # Avoid division by zero: if nr is 0, a is -t*lam < 0, Act_set is 0.
    # Result should be 0.
    # We can use np.divide with where, or simple mask

    scaling = np.zeros_like(nr)
    mask = (nr > 0).flatten()

    # Calculate for non-zero norms
    # (1 - t * lambda ./ nr)
    scaling[mask] = 1 - t * lam / nr[mask]

    output = Act_set * scaling * X
    return output


def solver(
    fhandle,
    gfhandle,
    fcalA,
    fcalAstar,
    fprox,
    fcalJ,
    x0,
    L,
    tol,
    upsilon,
    beta,
    mu,
    theta,
    maxiter,
    gamma,
):
    err = float('inf')
    x1 = x0
    # x2 = x1 No, initialized later
    fs = []
    Ds = []

    f1, x1 = fhandle(x1, mu)
    gf1 = gfhandle(x1, mu)

    t = 1.0 / L

    # fs(iter + 1) -> fs[iter] in python
    fs.append(f1)

    n, p = x0.main.shape
    Vinitial = np.zeros(
        (p, p)
    )  # This Vinitial seems to be used for warm start in finddir? Wait, passed to finddir.
    totalbt = 0
    innertol = max(1e-13, min(1e-11, 1e-3 * np.sqrt(tol) * t**2))

    import time

    for iter_idx in range(maxiter):

        # record time
        print(f"Iteration {iter_idx+1}, f1={f1}, mu={mu}, innertol={innertol}")
        start_time = time.time()

        # [V, Vinitial, inneriter] = finddir(...)
        V, Vinitial, inneriter = finddir(
            x1, gf1, t, fcalA, fcalAstar, fprox, fcalJ, gamma, Vinitial, innertol
        )

        norm_V = np.linalg.norm(V, 'fro')
        if norm_V <= tol and mu <= tol:
            break

        if norm_V <= mu:
            mu = theta * mu
            f1, x1 = fhandle(x1, mu)
            gf1 = gfhandle(x1, mu)
            # fs(iter) = f1; Note: MATLAB iter starts 1. So this replaces last fs?
            # In MATLAB loop is 1:maxiter. `fs(iter) = f1`.
            # But initially `fs(iter+1)=f1` where iter=0. So `fs(1)=f1`.
            # Inside loop iter=1. `fs(1)=f1`. So it overwrites?
            # The MATLAB code:
            # iter=0; fs(iter+1)=f1;
            # for iter = 1:maxiter
            #    ...
            #    if ... fs(iter) = f1; else ... fs(iter) = f2;
            # So yes, it overwrites/appends.
            # In Python list, we have [f_initial].
            # Loop i=0. We want to ADD the new value or Update?
            # It seems it records the function value at the END of iteration i.
            # So we should append.
            fs.append(f1)

        else:
            alpha = 1.0
            x2 = R(x1, alpha * V)
            f2, x2 = fhandle(x2, mu)
            btiter = 0
            while f2 > f1 - upsilon * alpha * norm_V**2 and btiter < 3:
                alpha = alpha * beta
                x2 = R(x1, alpha * V)
                f2, x2 = fhandle(x2, mu)
                btiter += 1
                totalbt += 1

            fs.append(f2)
            if btiter == 3:
                innertol = max(innertol * 1e-2, 1e-20)

            gf2 = gfhandle(x2, mu)
            err = norm_V
            Ds.append(np.linalg.norm(V, 'fro'))

            x1 = x2
            f1 = f2
            gf1 = gf2
        end_time = time.time()
        print(f"  Time for iteration {iter_idx+1}: {end_time - start_time} seconds")

    return x1, fs, Ds


def R(x, eta):
    # [Q, R] = qr(x.main + eta, 0)
    # [U, ~, V] = svd(R)
    # output.main = Q * (U * V')

    Q_qr, R_qr = la.qr(x.main + eta, mode='economic')
    U, S, Vh = la.svd(R_qr, full_matrices=False)
    # MATLAB svd(R) -> U, S, V. R = U*S*V'.
    # Python svd(R) -> U, S, Vh. R = U*S*Vh. so Vh is V' in MATLAB.
    # MATLAB: U * V'. Python: U @ Vh.

    new_main = Q_qr @ (U @ Vh)
    return Variable(new_main)


def E(Lambda, BLambda, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, zeta):
    if BLambda is None or BLambda.size == 0:
        # BLambda = x - t * (gfx - fcalAstar(Lambda, x));
        # x is ndarray here (passed from finddir x=xx.main)
        BLambda = x - t * (gfx - fcalAstar(Lambda, x))

    DLambda = fprox(BLambda, t, zeta) - x
    ELambda = fcalA(DLambda, x)
    return ELambda


def GLd(Lambda, d, BLambda, Blocks, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, zeta):
    # GLambdad = t * fcalA(fcalJ(BLambda, fcalAstar(d, x), t, zeta), x);
    term1 = fcalAstar(d, x)
    term2 = fcalJ(BLambda, term1, t, zeta)
    GLambdad = t * fcalA(term2, x)
    return GLambdad


def finddir(xx, gfx, t, fcalA, fcalAstar, fprox, fcalJ, zeta, x0, innertol):
    x = xx.main
    lam_param = 0.2
    nu = 0.99
    tau = 0.1
    eta1 = 0.2
    eta2 = 0.75
    omega1 = 3
    omega2 = 5
    alpha = 0.1
    beta_param = 1 / alpha / 100
    n, p = x.shape

    z = x0  # x0 here is Vinitial from solver, size p x p

    # BLambda = x - t * (gfx - fcalAstar(z, x));
    # z is p x p (?) Wait.
    # In Solver: Vinitial = zeros(p, p).
    # In MATLAB: fcalAstar(Lambda, U) -> U * (Lambda + Lambda').
    # U is x (n x p). Lambda is p x p.
    # So fcalAstar returns n x p.
    # gfx is n x p. x is n x p.
    # So BLambda is n x p.
    BLambda = x - t * (gfx - fcalAstar(z, x))

    Fz = E(z, BLambda, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, zeta)
    nFz = np.linalg.norm(Fz, 'fro')

    nnls = 5
    xi = np.zeros(nnls)
    xi[-1] = nFz

    maxiter = 1000
    Blocks = None  # Blocks cell(p, 1) in MATLAB unused?

    times_iter = 0
    # Loop 0 to maxiter
    for times_iter in range(maxiter + 1):
        if not np.isfinite(nFz):
            break

        iota = lam_param * max(min(nFz, 0.1), 1e-11)

        # Axhandle = @(d) GLd(...) + iota * d
        Axhandle = (
            lambda d: GLd(
                z, d, BLambda, Blocks, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, zeta
            )
            + iota * d
        )

        # d, CGiter = myCG(Axhandle, -Fz, tau, lam_param * nFz, 30)
        # Use scipy.sparse.linalg.cg for acceleration
        b_vec = -Fz.flatten()

        def ax_flat(v_vec):
            v_mat = v_vec.reshape(p, p)
            return Axhandle(v_mat).flatten()

        linear_op = spla.LinearOperator((p * p, p * p), matvec=ax_flat, dtype=float)
        # Use a fixed tolerance or relative tolerance appropriate for Newton-CG
        d_vec, _ = spla.cg(linear_op, b_vec, rtol=1e-5, maxiter=30)
        d = d_vec.reshape(p, p)
        CGiter = 30

        u = z + d
        Fu = E(u, None, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, zeta)
        nFu = np.linalg.norm(Fu, 'fro')

        if nFu < nu * np.max(xi):
            z = u
            Fz = Fu
            nFz = nFu
            xi[(times_iter) % nnls] = nFz
            status = 'success'
        else:
            rho_val = -np.sum(Fu * d) / (np.linalg.norm(d, 'fro') ** 2 + 1e-20)
            # Avoid div by zero

            if rho_val >= eta1:
                v = z - np.sum(Fu * (z - u)) / (nFu**2 + 1e-20) * Fu
                Fv = E(v, None, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, zeta)
                nFv = np.linalg.norm(Fv, 'fro')
                if nFv <= nFz:
                    z = v
                    Fz = Fv
                    nFz = nFv
                    status = 'safegard success projection'
                else:
                    z = z - beta_param * Fz
                    Fz = E(z, None, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, zeta)
                    nFz = nFz  # MATLAB updates nFz? Yes: nFz = norm(Fz, 'fro');
                    nFz = np.linalg.norm(Fz, 'fro')
                    status = 'safegard success fixed-point'
            else:
                status = 'safegard unsuccess'

            if rho_val >= eta2:
                lam_param = max(lam_param / 4, 1e-5)
            elif rho_val >= eta1:
                lam_param = (1 + omega1) / 2 * lam_param
            else:
                lam_param = (omega1 + omega2) / 2 * lam_param

        BLambda = x - t * (gfx - fcalAstar(z, x))

        if nFz**2 <= innertol:
            break

    Lambda = z
    inneriter = times_iter
    output = fprox(BLambda, t, zeta) - x
    return output, Lambda, inneriter


def calA(Z, U):  # U \in St(p, n)
    # tmp = Z' * U
    # output = tmp + tmp'
    # Z is n x p (Direction in tangent space? No, Z is DLambda).
    # DLambda = fprox(...) - x. Both n x p.
    # U is x. n x p.
    # Z' * U -> p x p.
    tmp = Z.T @ U
    output = tmp + tmp.T
    return output


def calAstar(Lambda, U):  # U \in St(p, n)
    # output = U * (Lambda + Lambda')
    # Lambda is p x p (Lagrange multiplier dual var?)
    output = U @ (Lambda + Lambda.T)
    return output


def calJ(y, eta, t, gamma):
    # output = (abs(y) > gamma * t) .* eta;
    return (np.abs(y) > gamma * t).astype(float) * eta


def myCG(Axhandle, b, tau, lambdanFz, maxiter):
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    k = 0

    # condition: norm(r) > tau * min(lambdanFz * norm(x), 1)
    # Note: norm(x) is 0 initially. min(0,1) = 0.
    # So if norm(r) > 0, it enters.

    while k < maxiter:
        cond_rhs = tau * min(lambdanFz * np.linalg.norm(x, 'fro'), 1)
        if np.linalg.norm(r, 'fro') <= cond_rhs:
            break

        Ap = Axhandle(p)

        # alpha = r' * r / (p' * Ap)
        rr = np.sum(r * r)
        pAp = np.sum(p * Ap)

        if pAp == 0:
            break

        alpha = rr / pAp
        x = x + alpha * p
        r_new = r - alpha * Ap

        rr_new = np.sum(r_new * r_new)
        beta = rr_new / rr

        p = r_new + beta * p
        r = r_new
        k += 1

    return x, k
