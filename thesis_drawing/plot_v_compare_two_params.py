'''
画两组参数下的 v 曲线，风格与现有脚本保持一致
'''

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 固定参数
tau_in = 0.4
tau_out = 10
tau_close = 100
tau_open = 130
v_crit = 0.13
n_gate = 0.1


def J_stim(t):
    return 0.05 if 40 <= t <= 50 else 0.0


def odes(t, y, tau_open, v_rest, v_peak):
    v, h = y

    J_in = (h * (v_peak - v) * (v - v_rest) ** 2) / tau_in
    J_out = -(v - v_rest) / tau_out
    dv_dt = J_in + J_out + J_stim(t)

    h_inf = 0.5 * (1 - math.tanh((v - v_crit) / n_gate))
    dh_dt = (1 / tau_close + (tau_close - tau_open) / tau_open / tau_close * h_inf) * (
        h_inf - h
    )

    return [dv_dt, dh_dt]


def simulate_v_curve(v_rest, v_peak):
    v0 = v_rest + 0.001
    h0 = 1.0
    t_span = (0, 300)
    t_eval = np.linspace(t_span[0], t_span[1], t_span[1] * 10 + 1)

    sol = solve_ivp(
        odes,
        t_span,
        [v0, h0],
        t_eval=t_eval,
        args=(tau_open, v_rest, v_peak),
    )
    return sol.t, sol.y[0]


def main():
    # 与示例图一致的两组参数
    t1, v1 = simulate_v_curve(v_rest=0.2, v_peak=0.8)
    t2, v2 = simulate_v_curve(v_rest=0.0, v_peak=1.0)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        t1,
        v1,
        color="tab:red",
        linewidth=2,
        label=r"$v_{peak}=0.8, v_{rest}=0.2$",
    )
    ax.plot(
        t2,
        v2,
        color="tab:blue",
        linewidth=2,
        label=r"$v_{peak}=1, v_{rest}=0$",
    )

    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_ylabel("v", fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(True)
    ax.legend(loc="upper right", fontsize=14)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
