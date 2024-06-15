import os
import yaml
import pymap3d
import numpy as np
import matplotlib.pyplot as plt

def parse_config(filename):
    """Parse JSBSim config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return type('EnvConfig', (object,), config_data)


def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')


def LLA2NEU(lon, lat, alt, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from Geodetic Coordinate System to NEU Coordinate System.

    Args:
        lon, lat, alt (float): target geodetic lontitude(°), latitude(°), altitude(m)
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (North, East, Up), unit: m
    """
    n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.array([n, e, -d])


def NEU2LLA(n, e, u, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from NEU Coordinate System to Geodetic Coordinate System.

    Args:
        n, e, u (float): target relative position w.r.t. North, East, Down
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (lon, lat, alt), unit: °, °, m
    """
    lat, lon, h = pymap3d.ned2geodetic(n, e, -u, lat0, lon0, alt0)
    return np.array([lon, lat, h])


def get_AO_TA_R(ego_feature, enm_feature, return_side=False):
    """Get AO & TA angles and relative distance between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): ego_AO, ego_TA, R
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y, delta_z])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag


def get2d_AO_TA_R(ego_feature, enm_feature, return_side=False):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag


def in_range_deg(angle):
    """ Given an angle in degrees, normalises in (-180, 180] """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def in_range_rad(angle):
    """ Given an angle in rads, normalises in (-pi, pi] """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

# 计算四元数q
def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to Quaternion.
    :param roll: roll angle (rad)
    :param pitch: pitch angle (rad)
    :param yaw: yaw angle (rad)
    :return: Quaternion (w, x, y, z)
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

# 计算四元数的时间导数 0.5 * ​q⊗ω q=(w,x,y,z)
# ω=(0,ωx,ωy,ωz)
# ⊗ 表示四元数乘法
def quaternion_derivative(q, omega):
    """
    Calculate the derivative of the quaternion.
    :param q: Quaternion (w, x, y, z)
    :param omega: Angular velocity (roll rate, pitch rate, yaw rate)
    :return: Quaternion derivative (dq/dt)
    """
    w, x, y, z = q
    wx, wy, wz = omega

    dqdt = 0.5 * np.array([
        -x * wx - y * wy - z * wz,
         w * wx + y * wz - z * wy,
         w * wy - x * wz + z * wx,
         w * wz + x * wy - y * wx
    ])

    return dqdt

def calculate_ossm(rpy, rpy_velocity):
    """
    Calculate the One-Step Smoothness Metric (OSSM).
    :param rpy: Tuple of (roll, pitch, yaw) in radians
    :param rpy_velocity: Tuple of (roll rate, pitch rate, yaw rate) in radians per second
    :return: OSSM value
    """
    roll, pitch, yaw = rpy
    roll_rate, pitch_rate, yaw_rate = rpy_velocity
    
    # 将姿态角转换为四元数
    q = euler_to_quaternion(roll, pitch, yaw)
    
    # 计算四元数的时间导数
    omega = np.array([roll_rate, pitch_rate, yaw_rate])
    dqdt = quaternion_derivative(q, omega)
    
    # 计算四元数时间导数的二范数，即OSSM
    ossm = np.linalg.norm(dqdt)
    
    return ossm

# 保存图像的函数
def save_ossm_plot(timestamps, ossm_values, save_dir='png'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 自动生成文件名后缀
    file_list = os.listdir(save_dir)
    existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in file_list if f.startswith('ossm_plot_') and f.endswith('.png')]
    new_index = max(existing_indices) + 1 if existing_indices else 0
    
    plt.figure()
    plt.plot(timestamps, ossm_values, 'r-')
    plt.xlabel('Timestamp')
    plt.ylabel('OSSM')
    plt.title('OSSM Over Time')
    filename = os.path.join(save_dir, f"ossm_plot_{new_index}.png")
    # 保存图像
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")
    
