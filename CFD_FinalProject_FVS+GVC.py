# %% 库

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# %% 基础参数设置

nx = 1000               # 网格数
gamma = 1.4             # 气体比热比
t_final = 0.5           # 最终时间
CFL = 0.5               # CFL数
xmin, xmax = -5.0, 5.0  # 计算域

# %% 初始条件

def initial_condition(x):
    rho = np.where(x < 0, 1.0, 0.125)  # 密度
    u = np.zeros_like(x)                # 速度
    p = np.where(x < 0, 1.0, 0.1)       # 压力
    return rho, u, p

# %% 守恒变量转原始变量

def conserved_to_primitive(U):
    # 保护性检查，防止负密度
    rho = np.maximum(U[0], 1e-10)
    u = U[1] / rho
    energy = U[2]
    # 确保动能不超过总能量
    kinetic_energy = 0.5 * rho * u**2
    internal_energy = np.maximum(energy - kinetic_energy, 1e-10)
    p = (gamma - 1) * internal_energy
    return rho, u, p

# %% 三阶Runge-Kutta时间推进

def rk3_step(U, dt, f):
    k1 = f(U)
    k2 = f(U + dt * k1)
    k3 = f(U + dt * (0.25*k1 + 0.25*k2))
    U_new = U + dt * (1/6*k1 + 1/6*k2 + 2/3*k3)
    return U_new

# %% Van Leer通量向量分裂

def flux_fvs_vanleer(U):
    # 获取原始变量
    rho = U[0, :]
    u = U[1, :] / rho
    p = (gamma - 1) * (U[2, :] - 0.5 * rho * u**2)
    
    # 保护性检查
    rho = np.maximum(rho, 1e-10)
    p = np.maximum(p, 1e-10)
    
    # 计算声速和马赫数
    a = np.sqrt(np.maximum(gamma * p / rho, 1e-10))
    M = u / np.maximum(a, 1e-10)
    
    # 总通量
    F = np.zeros_like(U)
    F[0, :] = rho * u
    F[1, :] = rho * u**2 + p
    F[2, :] = u * (U[2, :] + p)  # U[2]是总能E = p/(gamma-1)+0.5*rho*u^2
    
    # 初始化正负通量
    F_plus = np.zeros_like(U)
    F_minus = np.zeros_like(U)
    
    # 计算分裂通量
    for i in range(len(rho)):
        # 超音速右流 (M ≥ 1)
        if M[i] >= 1:
            F_plus[:, i] = F[:, i]
            F_minus[:, i] = 0.0
        
        # 亚音速流 (|M| < 1)
        elif abs(M[i]) < 1:
            # 正通量分量
            factor_plus = rho[i] * a[i] * (M[i] + 1)**2 / 4.0
            F_plus[0, i] = factor_plus
            F_plus[1, i] = factor_plus * (2 * a[i] / gamma + u[i] * (gamma - 1) / gamma)
            F_plus[2, i] = factor_plus * ((2 * a[i] / gamma + u[i] * (gamma - 1) / gamma)**2 / (2 * (gamma - 1)))
            
            # 负通量分量
            factor_minus = -rho[i] * a[i] * (M[i] - 1)**2 / 4.0
            F_minus[0, i] = factor_minus
            F_minus[1, i] = factor_minus * (-2 * a[i] / gamma + u[i] * (gamma - 1) / gamma)
            F_minus[2, i] = factor_minus * ((-2 * a[i] / gamma + u[i] * (gamma - 1) / gamma)**2 / (2 * (gamma - 1)))
        
        # 超音速左流 (M ≤ -1)
        else:  # M[i] <= -1
            F_plus[:, i] = 0.0
            F_minus[:, i] = F[:, i]
    
    return F_plus, F_minus

# %% GVC重构函数（群速度控制）

def gvc_reconstruction(v, direction='plus'):
    n = len(v)
    v_recon = np.zeros_like(v)
    eps = 1e-12  # 防止除零的小常数
    
    for i in range(n):
        # 边界处理（一阶外推）
        if i == 0 or i == n-1:
            v_recon[i] = v[i]
            continue
        
        # 计算梯度比 r
        if direction == 'plus':
            # 右界面重构 (i+1/2)
            numerator = v[i] - v[i-1]
            denominator = v[i+1] - v[i]
        else:  # direction == 'minus'
            # 左界面重构 (i-1/2)
            numerator = v[i] - v[i+1]
            denominator = v[i-1] - v[i]
        
        # 防止除零
        if np.abs(denominator) < eps:
            r = 0.0
        else:
            r = numerator / denominator
        
        # GVC限制器函数
        if np.abs(r) > 1:
            phi = 1.0
        else:
            phi = r
        
        # 重构界面值（添加限制器）
        if direction == 'plus':
            delta = phi * (v[i+1] - v[i]) / 2
            # 限制变化量，防止过大
            max_delta = 0.5 * (np.max(v[i-1:i+2]) - np.min(v[i-1:i+2]))
            delta = np.clip(delta, -max_delta, max_delta)
            v_recon[i] = v[i] + delta
        else:  # direction == 'minus'
            delta = phi * (v[i] - v[i-1]) / 2
            # 限制变化量，防止过大
            max_delta = 0.5 * (np.max(v[i-1:i+2]) - np.min(v[i-1:i+2]))
            delta = np.clip(delta, -max_delta, max_delta)
            v_recon[i] = v[i] - delta
    
    return v_recon

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

# %% 物理约束检查函数

def enforce_physical_constraints(U):
    # 检查密度
    rho = U[0, :]
    if np.any(rho <= 0):
        print(f"Warning: {np.sum(rho <= 0)} cells with non-positive density corrected.")
        rho = np.maximum(rho, 1e-10)
        U[0, :] = rho
    
    # 重新计算原始变量
    rho, u, p = conserved_to_primitive(U)
    
    # 检查压力
    if np.any(p <= 0):
        print(f"Warning: {np.sum(p <= 0)} cells with non-positive pressure corrected.")
        p = np.maximum(p, 1e-10)
        # 重新计算能量
        energy = p / (gamma - 1) + 0.5 * rho * u**2
        U[2, :] = energy
    
    return U

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
    a = np.sqrt(np.maximum(gamma * p / rho, 1e-10))  # 保护声速计算
    max_speed = np.max(np.abs(u) + a)
    dt = CFL * dx / (max_speed + 1e-10)  # 防止除零
    dt = min(dt, t_final - t)
    
    # 使用GVC重构
    UL_recon = np.zeros_like(U)
    UR_recon = np.zeros_like(U)
    
    # 对每个守恒变量分量分别进行GVC重构
    for comp in range(3):
        v = U[comp, :]
        
        # 重构左界面值 (i-1/2) - 使用minus方向
        vL = gvc_reconstruction(v, direction='minus')
        
        # 重构右界面值 (i+1/2) - 使用plus方向
        vR = gvc_reconstruction(v, direction='plus')
        
        UL_recon[comp, :] = vL
        UR_recon[comp, :] = vR
    
    # 计算通量（每个界面一个通量，共 nx+1 个）
    F = np.zeros((3, nx + 1))
    
    # 内部界面
    for i in range(1, nx):
        # 左单元右界面状态
        UL = UR_recon[:, i-1]
        # 右单元左界面状态
        UR = UL_recon[:, i]
        
        # 计算左右状态的通量分裂
        F_plus_L, F_minus_L = flux_fvs_vanleer(UL.reshape(3, 1))
        F_plus_R, F_minus_R = flux_fvs_vanleer(UR.reshape(3, 1))
        
        # FVS通量 = F^+(左) + F^-(右)
        F[:, i] = F_plus_L.flatten() + F_minus_R.flatten()
    
    # 边界处理
    # 左边界 (i=0)
    F_plus_L, F_minus_L = flux_fvs_vanleer(U[:, 0].reshape(3, 1))
    F_plus_R, F_minus_R = flux_fvs_vanleer(U[:, 0].reshape(3, 1))
    F[:, 0] = F_plus_L.flatten() + F_minus_R.flatten()
    
    # 右边界 (i=nx)
    F_plus_L, F_minus_L = flux_fvs_vanleer(U[:, -1].reshape(3, 1))
    F_plus_R, F_minus_R = flux_fvs_vanleer(U[:, -1].reshape(3, 1))
    F[:, -1] = F_plus_L.flatten() + F_minus_R.flatten()
    
    # 通量差分
    dF = (F[:, 1:] - F[:, :-1]) / dx
    
    # RK3时间推进
    U = rk3_step(U, dt, lambda U: -dF)
    
    # 确保物理量合理
    U = enforce_physical_constraints(U)
    
    # 更新时间
    t += dt
    iteration += 1
    print(f"Iteration: {iteration}, Time: {t:.4f}, dt: {dt:.6f}, Max density: {np.max(U[0]):.4f}, Min density: {np.min(U[0]):.4f}")

# 计算精确解
x_exact = np.linspace(xmin, xmax, 1000)
exact_rho, exact_u, exact_p = sod_exact(x_exact, t_final)

# 从守恒变量获取数值解
rho_num = U[0, :]
u_num = U[1, :] / rho_num
p_num = (gamma - 1) * (U[2, :] - 0.5 * rho_num * u_num**2)


# %% 绘图比较

plt.figure(figsize=(18, 12))
plt.suptitle(f'Sod Shock Tube Problem (t = {t_final}s)\nFVS (Van Leer) + GVC Group Velocity Control', 
             fontsize=24, fontweight='bold')

# 密度图
plt.subplot(311)
plt.plot(x, rho_num, 'b-', linewidth=2, label='Numerical (FVS+GVC)')
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