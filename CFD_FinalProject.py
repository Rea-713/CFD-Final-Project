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
    H_roe = (sqrt_rhoL * hL + sqrt_rhoR * hR) / (sqrt_rhoL + sqrt_rhoR) # 焓
    rho_roe = (0.5 * (sqrt_rhoL + sqrt_rhoR))**2                        # 密度
    a_roe = np.sqrt((gamma-1)*(H_roe - 0.5*u_roe**2))                   # 声速（通过焓推导）
    
    # 特征速度（通过解Jacobi矩阵得到）
    lambda1 = u_roe - a_roe                                             # 左行声波
    lambda2 = u_roe                                                     # 熵波
    lambda3 = u_roe + a_roe                                             # 右行声波
    
    # 熵修正
    eps = 1e-6
    lambda1 = np.where(np.abs(lambda1) < eps, (lambda1**2 + eps**2)/(2*eps), lambda1)
    lambda3 = np.where(np.abs(lambda3) < eps, (lambda3**2 + eps**2)/(2*eps), lambda3)
    
    lambda_abs = np.array([np.abs(lambda1), np.abs(lambda2), np.abs(lambda3)])  # 特征值绝对值向量
    
    # 通量差分裂
    delta_U = UR - UL
    F_L = np.array([rhoL*uL, rhoL*uL**2 + pL, uL*(UL[2] + pL)])
    F_R = np.array([rhoR*uR, rhoR*uR**2 + pR, uR*(UR[2] + pR)])

    # 右特征向量矩阵 R (列向量)
    R = np.zeros((3, 3))
    R[0, :] = [1, 1, 1]                                                        # 第一行: 密度分量
    R[1, :] = [u_roe - a_roe, u_roe, u_roe + a_roe]                            # 第二行: 速度分量
    R[2, :] = [H_roe - u_roe*a_roe, 0.5*u_roe**2, H_roe + u_roe*a_roe]         # 第三行: 能量分量
    
    # 左特征向量矩阵 L (行向量) = R^{-1}
    b1 = 0.5 * (gamma - 1) * u_roe**2 / a_roe**2
    b2 = (gamma - 1) / a_roe**2
    
    L = np.zeros((3, 3))
    L[0, :] = [0.5*(b1 + u_roe/a_roe), -0.5*(b2*u_roe + 1/a_roe), 0.5*b2]
    L[1, :] = [1 - b1, b2*u_roe, -b2]
    L[2, :] = [0.5*(b1 - u_roe/a_roe), -0.5*(b2*u_roe - 1/a_roe), 0.5*b2]
    
    delta_U = UR - UL                                                          # 守恒变量差
    
    # 波强度 α = L · ΔU
    alpha = L @ delta_U
    
    # 耗散项计算： |Λ|α
    abs_lambda_alpha = lambda_abs * alpha
    
    # 耗散向量 = R(|Λ|α)
    diss_vector = R @ abs_lambda_alpha
    
    # 左右物理通量
    F_L = np.array([
        rhoL * uL, 
        rhoL * uL**2 + pL, 
        uL * (UL[2] + pL)])
    
    F_R = np.array([
        rhoR * uR, 
        rhoR * uR**2 + pR, 
        uR * (UR[2] + pR)])
    
    # Roe 通量公式
    F_roe = 0.5 * (F_L + F_R) - 0.5 * diss_vector
    
    return F_roe


# %% WENO

def weno(v): # 五阶
    n = len(v)
    vL = np.zeros_like(v)
    vR = np.zeros_like(v)
    epsilon = 1e-6

    for i in range(2, n-2):
        # 右重构
        v0 = v[i-2]
        v1 = v[i-1]
        v2 = v[i]
        v3 = v[i+1]
        v4 = v[i+2]

        # 光滑指示器（利用公式）
        IS0 = 13/12*(v0 - 2*v1 + v2)**2 + 1/4*(v0 - 4*v1 + 3*v2)**2
        IS1 = 13/12*(v1 - 2*v2 + v3)**2 + 1/4*(v1 - v3)**2
        IS2 = 13/12*(v2 - 2*v3 + v4)**2 + 1/4*(3*v2 - 4*v3 + v4)**2

        # 权重（泰勒展开）
        alpha0 = 0.1 / (IS0 + epsilon)**2
        alpha1 = 0.6 / (IS1 + epsilon)**2
        alpha2 = 0.3 / (IS2 + epsilon)**2
        sum_alpha = alpha0 + alpha1 + alpha2
        w0 = alpha0 / sum_alpha
        w1 = alpha1 / sum_alpha
        w2 = alpha2 / sum_alpha

        # 重构右界面值
        vR[i] = w0*( (2*v0 - 7*v1 + 11*v2)/6 ) + w1*( (-v1 +5*v2 +2*v3)/6 ) + w2*( (2*v2 +5*v3 -v4)/6 )

        # 左重构（对称处理）
        v0 = v[i+2]
        v1 = v[i+1]
        v2 = v[i]
        v3 = v[i-1]
        v4 = v[i-2]

        IS0 = 13/12*(v0 - 2*v1 + v2)**2 + 1/4*(v0 - 4*v1 +3*v2)**2
        IS1 = 13/12*(v1 -2*v2 +v3)**2 +1/4*(v1 -v3)**2
        IS2 = 13/12*(v2 -2*v3 +v4)**2 +1/4*(3*v2 -4*v3 +v4)**2

        alpha0 = 0.1/(IS0 + epsilon)**2
        alpha1 = 0.6/(IS1 + epsilon)**2
        alpha2 = 0.3/(IS2 + epsilon)**2
        sum_alpha = alpha0 + alpha1 + alpha2
        w0 = alpha0 / sum_alpha
        w1 = alpha1 / sum_alpha
        w2 = alpha2 / sum_alpha

        vL[i] = w0*( (2*v0 -7*v1 +11*v2)/6 ) + w1*( (-v1 +5*v2 +2*v3)/6 ) + w2*( (2*v2 +5*v3 -v4)/6 )

    return vL, vR

# %% Sod精确解（Riemann求解器）

def sod_exact(x, t):
    # 初始左右状态
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    gamma = 1.4



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



