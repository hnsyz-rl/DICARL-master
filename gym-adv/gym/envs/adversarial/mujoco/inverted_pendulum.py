import numpy as np
from gym import utils, spaces
from gym.envs.adversarial.mujoco import mujoco_env
import math

Theta_threshold_radians = 20 * 2 * math.pi / 360
X_threshold = 2

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        
        self.alive_bonus = 1.0

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)  # 这里的２设定了帧率，帧跳转
        ## Adversarial setup
        self._adv_f_bname = b'pole' #Byte String name of body on which the adversary force will be applied
        # 字节字符串，将在其上施加对抗力量的主体，是指在哪个位置施力，word\cart\pole这三个地方
        bnames = self.model.body_names  # 这里就是从mujoco中获取环境中包含的信息
        self._adv_bindex = bnames.index(self._adv_f_bname) #Index of the body on which the adversary force will be applied
        # 施加对抗力量的机构的索引，这里时索引２，也就是对车杆施力
        adv_max_force = 5.
        high_adv = np.ones(2)*adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space

    def _adv_to_xfrc(self, adv_act):  # 干扰力转换为笛卡尔力
        # print(self.model.data.xfrc_applied)
        new_xfrc = self.model.data.xfrc_applied*0.0  #　qfrc－generalized force总合力/ xfrc－cartesian force torque笛卡尔力
        new_xfrc[self._adv_bindex] = np.array([adv_act[0], 0., adv_act[1], 0., 0., 0.])
        self.model.data.xfrc_applied = new_xfrc

    def sample_action(self):
        class act(object):
            def __init__(self,pro=None,adv=None):
                self.pro=pro
                self.adv=adv
        sa = act(self.pro_action_space.sample(), self.adv_action_space.sample())
        return sa

    def _step(self, action):
        if hasattr(action, '__dict__'):
            self._adv_to_xfrc(action.adv)
            a = action.pro
        else:
            a = action

        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        # print('qfrc_passive=',self.model.data.qfrc_passive)
        # print('qfrc_applied=',self.model.data.qfrc_applied)
        # print('xfrc_inverse=',self.model.data.xfrc_applied)
        # print('qfrc_actuator=',self.model.data.qfrc_actuator)
        # print('a', a)

        ob = self._get_obs() # * [20, 1, 20, 1]
        x, theta, x_dot, theta_dot = ob
        # print(ob)
        # cost = 1* (x)**2/100 + 15 *(theta/ Theta_threshold_radians)**2 - 1
        cost = 15 *(theta/ Theta_threshold_radians)**2 - self.alive_bonus
        # notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= Theta_threshold_radians)

        notdone = np.abs(ob[0])<= X_threshold and (np.abs(ob[1]) <= Theta_threshold_radians)

        done = not notdone
        
        # print('ob', ob)
        # print('acc', self.model.data.qacc)
        #
        # if done:
        #     print('out:', np.isfinite(ob).all(), 'fail', (np.abs(ob[1]) <= Theta_threshold_radians))
        #     print('x', ob[0], 'theata', ob[1], 'cost', cost)
        return ob, cost, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)  # 设定随机位置
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)  # 设定随机速度
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_model_zero(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()  # data.qpos为位置与角度；data.qvel为对应加速度

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid=0
        v.cam.distance = v.model.stat.extent
