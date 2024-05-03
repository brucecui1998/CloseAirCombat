import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading


class HeadingTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            HeadingReward(self.config), # 航向奖励
            AltitudeReward(self.config), # 高度奖励
        ]
        self.termination_conditions = [
            UnreachHeading(self.config), # 未达到预定航向
            ExtremeState(self.config), # 极端状态
            Overload(self.config), # 过载
            LowAltitude(self.config), # 高度过低
            Timeout(self.config), # 超时
        ]

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        self.state_var = [
            c.delta_altitude,                   # 0. delta_h   (unit: m)
            c.delta_heading,                    # 1. delta_heading  (unit: °)
            c.delta_velocities_u,               # 2. delta_v   (unit: m/s)
            c.position_h_sl_m,                  # 3. altitude  (unit: m)
            c.attitude_roll_rad,                # 4. roll      (unit: rad)
            c.attitude_pitch_rad,               # 5. pitch     (unit: rad)
            c.velocities_u_mps,                 # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 8. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                # 9. vc        (unit: m/s)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.] 副翼
            c.fcs_elevator_cmd_norm,            # [-1., 1.] 升降舵
            c.fcs_rudder_cmd_norm,              # [-1., 1.] 方向舵
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9] 油门
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(12,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.
        进行了适当的单位转换和归一化处理，将高度从米转换为千米，将航向从度转换为弧度，将速度进行无量纲化处理
        使用三角函数处理了姿态的角度

        observation(dim 12):
            0. ego delta altitude      (unit: km)
            1. ego delta heading       (unit rad)
            2. ego delta velocities_u  (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego v_body_x            (unit: mh)
            9. ego v_body_y            (unit: mh)
            10. ego v_body_z           (unit: mh)
            11. ego_vc                 (unit: mh)
        """
        obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(12)
        norm_obs[0] = obs[0] / 1000         # 0. ego delta altitude (unit: 1km) 相对高度的变化量，单位从米转换为千米
        norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading  (unit rad) 相对航向的变化量，单位从度转换为弧度。
        norm_obs[2] = obs[2] / 340          # 2. ego delta velocities_u (unit: mh) 机体坐标系下的u轴（向前）速度变化量，已进行无量纲化处理
        norm_obs[3] = obs[3] / 5000         # 3. ego_altitude   (unit: 5km) 绝对高度，单位从米转换为每5千米
        norm_obs[4] = np.sin(obs[4])        # 4. ego_roll_sin 滚转角的正弦值
        norm_obs[5] = np.cos(obs[4])        # 5. ego_roll_cos 滚转角的余弦值
        norm_obs[6] = np.sin(obs[5])        # 6. ego_pitch_sin 俯仰角的正弦值
        norm_obs[7] = np.cos(obs[5])        # 7. ego_pitch_cos 俯仰角的余弦值
        norm_obs[8] = obs[6] / 340          # 8. ego_v_north    (unit: mh) 机体坐标系下的北向速度，已进行无量纲化处理 Mach"（马赫数），即以声速为单位的无量纲速度
        norm_obs[9] = obs[7] / 340          # 9. ego_v_east     (unit: mh) 机体坐标系下的东向速度，已进行无量纲化处理
        norm_obs[10] = obs[8] / 340         # 10. ego_v_down    (unit: mh) 机体坐标系下的下行速度，已进行无量纲化处理
        norm_obs[11] = obs[9] / 340         # 11. ego_vc        (unit: mh) 飞行器的总速度，已进行无量纲化处理
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high) #确保所有观测值都在这个预定义的空间内，可以避免因为极端或非法的状态值而导致学习算法表现不佳或者出现不稳定的情况
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """将离散动作索引转换为连续的动作值
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act
