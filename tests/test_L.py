import numpy as np
from scipy.optimize import minimize

# 定义目标函数
def objective(vars):
    x, y, lambda_ = vars
    # 拉格朗日函数
    value = -(x + y + lambda_ * (1 - x * x - y * y))
    return value

# 打印约束函数的值
def constraint(vars):
    lambda_ = vars[2]
    return lambda_

# 初始猜测值
initial_guess = [0.5, 0.5, 0.5]

# 设置约束条件：lambda >= 0
constraints = [
    {'type': 'ineq', 'fun': constraint}  # lambda >= 0
]

# 设置变量的边界条件，防止数值过大
bounds = [(-1, 1), (-1, 1), (0, None)]

# 执行优化，使用 SLSQP 方法
result = minimize(objective, initial_guess, constraints=constraints, bounds=bounds, method='SLSQP')

# 提取结果
x_opt, y_opt, lambda_opt = result.x
# 取反得到实际的最大值
max_value = -result.fun

print(f"Optimal x: {x_opt}")
print(f"Optimal y: {y_opt}")
print(f"Optimal lambda: {lambda_opt}")
print(f"Maximum value of objective function: {max_value}")
