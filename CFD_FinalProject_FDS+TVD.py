# %% 库

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# %% 基础参数设置

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

# %% 三阶Runge-Kutta时间推进

def rk3_step(U, dt, f):
    k1 = f(U)
    k2 = f(U + dt * k1)
    k3 = f(U + dt * (0.25*k1 + 0.25*k2))
    U_new = U + dt * (1/6*k1 + 1/6*k2 + 2/3*k3)
    return U_new

# %% Roe通量计算

def flux_roe(UL, UR):
    rhoL, uL, pL = conserved_to_primitive(UL)
    rhoR, uR, pR = conserved_to_primitive(UR)
    hL = (gamma/(gamma-1)) * pL/rhoL + 0.5*uL**2  # 左焓
    hR = (gamma/(gamma-1)) * pR/rhoR + 0.5*uR**2  # 右焓
    
    sqrt_rhoL = np.sqrt(rhoL)
    sqrt_rhoR = np.sqrt(rhoR)
    
    # Roe平均
    u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) / (sqrt_rhoL + sqrt_rhoR)  # 速度
    H_roe = (sqrt_rhoL * hL + sqrt_rhoR * hR) / (sqrt_rhoL + sqrt_rhoR)  # 焓
    a_roe = np.sqrt((gamma-1)*(H_roe - 0.5*u_roe**2))  # 声速
    
    # 特征速度
    lambda1 = u_roe - a_roe  # 左行声波
    lambda2 = u_roe          # 熵波
    lambda3 = u_roe + a_roe  # 右行声波
    
    # 熵修正
    eps = 1e-6
    lambda1 = np.where(np.abs(lambda1) < eps, (lambda1**2 + eps**2)/(2*eps), lambda1)
    lambda3 = np.where(np.abs(lambda3) < eps, (lambda3**2 + eps**2)/(2*eps), lambda3)
    
    lambda_abs = np.array([np.abs(lambda1), np.abs(lambda2), np.abs(lambda3)])  # 特征值绝对值
    
    # 通量差分裂
    delta_U = UR - UL
    
    # 右特征向量矩阵
    R = np.zeros((3, 3))
    R[0, :] = [1, 1, 1]
    R[1, :] = [u_roe - a_roe, u_roe, u_roe + a_roe]
    R[2, :] = [H_roe - u_roe*a_roe, 0.5*u_roe**2, H_roe + u_roe*a_roe]
    
    # 左特征向量矩阵
    b1 = 0.5 * (gamma - 1) * u_roe**2 / a_roe**2
    b2 = (gamma - 1) / a_roe**2
    
    L = np.zeros((3, 3))
    L[0, :] = [0.5*(b1 + u_roe/a_roe), -0.5*(b2*u_roe + 1/a_roe), 0.5*b2]
    L[1, :] = [1 - b1, b2*u_roe, -b2]
    L[2, :] = [0.5*(b1 - u_roe/a_roe), -0.5*(b2*u_roe - 1/a_roe), 0.5*b2]
    
    # 波强度
    alpha = L @ delta_U
    
    # 耗散项
    abs_lambda_alpha = lambda_abs * alpha
    diss_vector = R @ abs_lambda_alpha
    
    # 左右物理通量
    F_L = np.array([rhoL*uL, rhoL*uL**2 + pL, uL*(UL[2] + pL)])
    F_R = np.array([rhoR*uR, rhoR*uR**2 + pR, uR*(UR[2] + pR)])
    
    # Roe通量
    F_roe = 0.5 * (F_L + F_R) - 0.5 * diss_vector
    
    return F_roe

# %% Minmod限制器

def minmod(v, limiter='minmod'):
    n = len(v)
    vL = np.zeros_like(v)  # 左界面重构值
    vR = np.zeros_like(v)  # 右界面重构值
    
    # 边界处理（使用一阶重构）
    vL[0] = v[0]
    vR[0] = v[0]
    vL[-1] = v[-1]
    vR[-1] = v[-1]
    
    for i in range(1, n-1):
        # 计算斜率
        deltaL = v[i] - v[i-1]  # 左侧斜率
        deltaR = v[i+1] - v[i]  # 右侧斜率
        
        # 应用minmod限制器
        if deltaL * deltaR <= 0:
            slope = 0  # 符号不同，斜率为0
        else:
            # 取绝对值较小的斜率
            slope = np.sign(deltaL) * min(abs(deltaL), abs(deltaR))
        
        # 界面重构
        vL[i] = v[i] - 0.5 * slope
        vR[i] = v[i] + 0.5 * slope
    
    return vL, vR

# %% Sod精确解（Riemann求解器）

def sod_exact(x, t):
    # 初始左右状态
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    gamma = 1.4
    
    # 定义中间压力的Newton迭代函数
    def f(p_star):
        if p_star > pL:
            # 左激波
            AL = (p_star - pL) * np.sqrt((1 - (gamma-1)/(gamma+1)) / (rhoL * (p_star + (gamma-1)/(gamma+1)*pL)))
        else:
            # 左膨胀波
            AL = (2*aL)/(gamma-1) * ((p_star/pL)**((gamma-1)/(2*gamma)) - 1)
        
        if p_star > pR:
            # 右激波
            AR = (p_star - pR) * np.sqrt((1 - (gamma-1)/(gamma+1)) / (rhoR * (p_star + (gamma-1)/(gamma+1)*pR)))
        else:
            # 右膨胀波
            AR = (2*aR)/(gamma-1) * ((p_star/pR)**((gamma-1)/(2*gamma)) - 1)
        
        return AL + AR - (uR - uL)
    
    # 计算声速
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)
    
    # 使用牛顿迭代法求解中间压力
    p_initial = 0.5 * (pL + pR)
    try:
        p_star = newton(f, p_initial, maxiter=100, tol=1e-6)
    except:
        p_star = p_initial  # 若迭代失败，使用初始猜测
    
    # 计算中间速度
    if p_star > pL:
        # 左激波
        u_star = uL + (p_star - pL) * np.sqrt((1 - (gamma-1)/(gamma+1)) / (rhoL * (p_star + (gamma-1)/(gamma+1)*pL)))
    else:
        # 左膨胀波
        u_star = uL + (2*aL)/(gamma-1) * (1 - (p_star/pL)**((gamma-1)/(2*gamma)))
    
    # 计算波系结构
    # 左膨胀波尾部速度（如果存在）
    if p_star <= pL:
        a_starL = aL * (p_star/pL)**((gamma-1)/(2*gamma))
        u_exp_left = u_star - a_starL
    else:
        # 左激波速度
        u_shock_left = uL - aL * np.sqrt((gamma+1)/(2*gamma)*(p_star/pL - 1) + 1)
    
    # 右激波速度（如果存在）
    if p_star > pR:
        u_shock_right = uR + aR * np.sqrt((gamma+1)/(2*gamma)*(p_star/pR - 1) + 1)
    
    # 各区间的解
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)
    
    for i in range(len(x)):
        xi = x[i] / t if t != 0 else 0  # 避免除以零
        
        # 1. 左均匀区
        if p_star > pL:  # 左激波情况
            if xi <= u_shock_left:
                rho[i] = rhoL
                u[i] = uL
                p[i] = pL
            # 2. 激波后区域
            elif xi <= u_star:
                rho[i] = rhoL * ((gamma+1)*p_star + (gamma-1)*pL) / ((gamma-1)*p_star + (gamma+1)*pL)
                u[i] = u_star
                p[i] = p_star
        else:  # 左膨胀波情况
            # 1. 左均匀区
            if xi <= uL - aL:
                rho[i] = rhoL
                u[i] = uL
                p[i] = pL
            # 2. 膨胀波区
            elif xi <= u_exp_left:
                u[i] = (2/(gamma+1)) * (aL + (gamma-1)/2*uL + xi)
                a = aL - (gamma-1)/2*(u[i] - uL)
                rho[i] = rhoL * (a/aL)**(2/(gamma-1))
                p[i] = pL * (rho[i]/rhoL)**gamma
            # 3. 中间左均匀区
            elif xi <= u_star:
                rho[i] = rhoL * (p_star/pL)**(1/gamma)
                u[i] = u_star
                p[i] = p_star
        
        # 4. 中间右均匀区（接触间断右侧）
        if xi > u_star:
            if p_star > pR:  # 右激波情况
                # 4. 中间右均匀区（激波前）
                if xi <= u_shock_right:
                    rho[i] = rhoR * ((gamma+1)*p_star + (gamma-1)*pR) / ((gamma-1)*p_star + (gamma+1)*pR)
                    u[i] = u_star
                    p[i] = p_star
                # 5. 右均匀区
                else:
                    rho[i] = rhoR
                    u[i] = uR
                    p[i] = pR
            else:  # 右膨胀波情况
                # 4. 中间右均匀区（膨胀波前）
                a_starR = aR * (p_star/pR)**((gamma-1)/(2*gamma))
                u_exp_right = u_star + a_starR
                if xi <= u_exp_right:
                    rho[i] = rhoR * (p_star/pR)**(1/gamma)
                    u[i] = u_star
                    p[i] = p_star
                # 5. 膨胀波区
                elif xi <= uR + aR:
                    u[i] = (2/(gamma+1)) * (-aR + (gamma-1)/2*uR + xi)
                    a = aR + (gamma-1)/2*(u[i] - uR)
                    rho[i] = rhoR * (a/aR)**(2/(gamma-1))
                    p[i] = pR * (rho[i]/rhoR)**gamma
                # 6. 右均匀区
                else:
                    rho[i] = rhoR
                    u[i] = uR
                    p[i] = pR
    
    return rho, u, p

# %% 主程序

# 创建网格
x = np.linspace(xmin, xmax, nx)
dx = (xmax - xmin) / (nx - 1)

# 初始化变量
rho, u, p = initial_condition(x)
U = np.array([rho, rho * u, p / (gamma - 1) + 0.5 * rho * u**2])  # 守恒变量

t = 0
iteration = 0

# 时间推进循环
while t < t_final:
    # 计算时间步长
    rho, u, p = conserved_to_primitive(U)
    a = np.sqrt(gamma * p / rho)  # 声速
    max_speed = np.max(np.abs(u) + a)  # 最大波速
    dt = CFL * dx / max_speed
    dt = min(dt, t_final - t)  # 确保不会超过最终时间
    
    # Minmod限制器
    UL_recon = np.zeros_like(U)
    UR_recon = np.zeros_like(U)
    
    # 对每个守恒变量分量分别进行重构
    for i in range(3):
        v = U[i, :]
        vL, vR = minmod(v, limiter='minmod')
        UL_recon[i, :] = vL
        UR_recon[i, :] = vR
    
    # 计算通量（每个界面一个通量，共 nx+1 个）
    F = np.zeros((3, nx + 1))
    
    # 左边界
    F[:, 0] = flux_roe(UL_recon[:, 0], UR_recon[:, 0])
    
    # 右边界
    F[:, -1] = flux_roe(UL_recon[:, -1], UR_recon[:, -1])
    
    # 内部界面
    for i in range(1, nx):
        left_state = UR_recon[:, i - 1]  # 左单元的右界面值
        right_state = UL_recon[:, i]     # 右单元的左界面值
        F[:, i] = flux_roe(left_state, right_state)
    
    # 通量差分
    dF = (F[:, 1:] - F[:, :-1]) / dx
    
    # RK3时间推进
    U = rk3_step(U, dt, lambda U: -dF)
    
    # 更新时间
    t += dt
    iteration += 1
    print(f"Iteration: {iteration}, Time: {t:.4f}, dt: {dt:.6f}")

# 计算精确解
x_exact = np.linspace(xmin, xmax, 1000)
exact_rho, exact_u, exact_p = sod_exact(x_exact, t_final)

# 从守恒变量获取数值解
rho_num = U[0, :]
u_num = U[1, :] / rho_num
p_num = (gamma - 1) * (U[2, :] - 0.5 * rho_num * u_num**2)

# %%绘图比较

plt.figure(figsize=(18, 12))
plt.suptitle(f'Sod Shock Tube Problem (t = {t_final}s)\nFDS (Roe) + TVD (Minmod) ', 
             fontsize=24, fontweight='bold')

# 密度图
plt.subplot(311)
plt.plot(x, rho_num, 'b-', linewidth=2, label='Numerical (Minmod)')
plt.plot(x_exact, exact_rho, 'r--', linewidth=2, label='Exact')
plt.ylabel('Density', fontsize=16)
plt.legend(fontsize=14, loc='upper right')
plt.grid(alpha=0.3)
plt.title('Density Comparison', fontsize=18)

# 速度图
plt.subplot(312)
plt.plot(x, u_num, 'b-', linewidth=2, label='Numerical')
plt.plot(x_exact, exact_u, 'r--', linewidth=2, label='Exact')
plt.ylabel('Velocity', fontsize=16)
plt.legend(fontsize=14, loc='upper right')
plt.grid(alpha=0.3)
plt.title('Velocity Comparison', fontsize=18)

# 压力图
plt.subplot(313)
plt.plot(x, p_num, 'b-', linewidth=2, label='Numerical')
plt.plot(x_exact, exact_p, 'r--', linewidth=2, label='Exact')
plt.ylabel('Pressure', fontsize=16)
plt.xlabel('x', fontsize=16)
plt.legend(fontsize=14, loc='upper right')
plt.grid(alpha=0.3)
plt.title('Pressure Comparison', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局
plt.show()

    
    
    
    
    