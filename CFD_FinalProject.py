# %% 库

import numpy as np
import matplotlib.pyplot as plt

# %% 基础参数

nx = 1000               # 网格数
gamma = 1.4             # 气体比热比
t_final = 0.5           # 最终时间
CFL = 0.8               # CFL数
xmin, xmax = -5.0, 5.0  # 计算域

# %% 初始条件

def initial_condition(x):
    rho = np.where(x < 0, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.where(x < 0, 1.0, 0.1)
    return rho, u, p

# %% 守恒变量转原始变量

def conserved_to_primitive(U):
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1) * (U[2] - 0.5 * rho * u**2)
    return rho, u, p

# %%
