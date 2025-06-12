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
    p = (gamma - 1) * (U[2] - 0.5 * rho * u**2) # 利用理想气体状态方程
    return rho, u, p

# %% 三阶Runge-Kutta时间推进

def rk3_step(U, dt, f):
    k1 = f(U)                           
    k2 = f(U + dt * k1)                
    k3 = f(U + dt * (0.25*k1 + 0.25*k2))
    U_new = U + dt * (1/6*k1 + 1/6*k2 + 2/3*k3)  
    return U_new

# %% Roe通量计算

def flux_roe(UL, UR):
    
    # 左右原始变量
    rhoL, uL, pL = conserved_to_primitive(UL)   # 左常数状态
    rhoR, uR, pR = conserved_to_primitive(UR)   # 右常数状态
    hL = (gamma/(gamma-1)) * pL/rhoL + 0.5*uL**2  # 左焓
    hR = (gamma/(gamma-1)) * pR/rhoR + 0.5*uR**2  # 右焓
    
    sqrt_rhoL = np.sqrt(rhoL)
    sqrt_rhoR = np.sqrt(rhoR)
    
    # Roe平均
    u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) / (sqrt_rhoL + sqrt_rhoR) # 速度
    h_roe = (sqrt_rhoL * hL + sqrt_rhoR * hR) / (sqrt_rhoL + sqrt_rhoR) # 焓
    rho_roe = (0.5 * (sqrt_rhoL + sqrt_rhoR))**2                        # 密度
    a_roe = np.sqrt((gamma-1)*(h_roe - 0.5*u_roe**2))                   # 声速（通过焓推导）
    
    # 特征速度（通过解Jacobi矩阵得到）
    lambda1 = u_roe - a_roe
    lambda2 = u_roe
    lambda3 = u_roe + a_roe
    
    # 熵修正
    eps = 1e-6
    lambda1 = np.where(np.abs(lambda1) < eps, (lambda1**2 + eps**2)/(2*eps), lambda1)
    lambda3 = np.where(np.abs(lambda3) < eps, (lambda3**2 + eps**2)/(2*eps), lambda3)
    
    # 通量差分裂
    delta_U = UR - UL
    F_L = np.array([rhoL*uL, rhoL*uL**2 + pL, uL*(UL[2] + pL)])
    F_R = np.array([rhoR*uR, rhoR*uR**2 + pR, uR*(UR[2] + pR)])




# %% WENO

def weno(v):
    n = len(v)
    vL = np.zeros_like(v)
    vR = np.zeros_like(v)



# %% 

x = np.linspace(xmin, xmax, nx)
dx = (xmax - xmin)/(nx-1)
rho, u, p = initial_condition(x)
U = np.array([rho, rho*u, p/(gamma-1) + 0.5*rho*u**2])  # 形状 (3, nx)

t = 0

# %% 绘图

plt.figure(figsize=(18,10))


plt.legend()
plt.tight_layout()
plt.show()



