import time
import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID

# 模拟系统的输出函数
def system_output(current_value, control_input):
    # 简单模拟系统，新的输出值 = 当前值 + 控制输入
    new_value = current_value + control_input
    return new_value

# 初始化参数
setpoint = 10  # 目标值
current_value = 0  # 系统初始值
time_steps = 100  # 模拟的时间步数
dt = 0.1  # 每一步的时间间隔

# 创建PID控制器
#pid = PID(1, 0.1, 0.05, setpoint=setpoint) 效果不好
pid = PID(0.5, 0.05, 0.02, setpoint=setpoint)
pid.sample_time = dt  # 设置PID控制器的采样时间

# 用于存储结果的列表
output_values = []
control_inputs = []

# 运行PID控制器
for _ in range(time_steps):
    control_input = pid(current_value)  # 计算控制输入
    current_value = system_output(current_value, control_input)  # 计算系统的下一个输出值

    # 存储结果
    output_values.append(current_value)
    control_inputs.append(control_input)

    # 模拟时间延迟
    time.sleep(dt)

# 绘制结果
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(output_values, label='Output')
plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
plt.xlabel('Time step')
plt.ylabel('System Output')
plt.legend()
plt.title('PID Control System Output')

plt.subplot(2, 1, 2)
plt.plot(control_inputs, label='Control Input')
plt.xlabel('Time step')
plt.ylabel('Control Input')
plt.legend()
plt.title('PID Control Input')

plt.tight_layout()
plt.savefig("demo.png")
