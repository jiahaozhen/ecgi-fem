'''
测试 不同tau_open对 APD 的影响
结论 tau_open 越大 APD 越短
'''

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 固定参数
tau_in = 0.4
tau_out = 10
tau_close = 100
tau_open = 130
v_crit = 0.13
v_rest = 0
v_peak = 1


def J_stim(t):
    return 0.01 if 40 <= t <= 50 else 0.0


def odes(t, y, tau_open):
    v, h = y

    # 电流项
    J_in = (h * (v_peak - v) * (v - v_rest) ** 2) / tau_in
    J_out = -(v - v_rest) / tau_out

    dv_dt = J_in + J_out + J_stim(t)

    # 门控变量动力学
    n_gate = 0.1
    h_inf = 0.5 * (1 - math.tanh((v - v_crit) / n_gate))
    dh_dt = (1 / tau_close + (tau_close - tau_open) / tau_open / tau_close * h_inf) * (
        h_inf - h
    )

    return [dv_dt, dh_dt]


# 初始条件与时间设置
v0 = v_rest + 0.001
h0 = 1.0
t_span = (0, 300)
t_eval = np.linspace(t_span[0], t_span[1], t_span[1] * 10 + 1)


sol = solve_ivp(odes, t_span, [v0, h0], t_eval=t_eval, args=(tau_open,))
t = sol.t
v = sol.y[0]
h = sol.y[1]

fig, ax_v = plt.subplots(figsize=(9, 5))
ax_h = ax_v.twinx()

line_v = ax_v.plot(t, v, color="tab:blue", linewidth=2, label="v")[0]
line_h = ax_h.plot(t, h, color="tab:red", linewidth=2, linestyle="--", label="h")[0]

ax_v.set_xlabel("Time (ms)", fontsize=14)
ax_v.set_ylabel("v (Membrane potential)", fontsize=14, color="tab:blue")
ax_h.set_ylabel("h (Gating variable)", fontsize=14, color="tab:red")
ax_v.tick_params(axis="x", labelsize=14)
ax_v.tick_params(axis="y", labelsize=14, labelcolor="tab:blue")
ax_h.tick_params(axis="y", labelsize=14, labelcolor="tab:red")
ax_v.tick_params(axis="y", labelcolor="tab:blue")
ax_h.tick_params(axis="y", labelcolor="tab:red")
ax_v.grid(True)

ax_v.legend([line_v, line_h], ["v", "h"], loc="upper right", fontsize=14)
# ax_v.set_title(f"v and h curves ($\\tau_{{open}}={tau_open}$)", fontsize=14)

fig.tight_layout()
plt.show()
