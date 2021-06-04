import numpy as np
from gym import utils,spaces
from gym.envs.adversarial.mujoco import mujoco_env

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.des_v = 3.5
        self.alive_bonus = 20.0
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)
        ## Adversarial setup
        self._adv_f_bname = [b'torso', b'thigh'] #Byte String name of body on which the adversary force will be applied  # 躯干施力
        bnames = self.model.body_names
        self._adv_bindex = [bnames.index(i) for i in self._adv_f_bname] #Index of the body on which the adversary force will be applied
        adv_max_force = 5.0
        high_adv = np.ones(2*len(self._adv_bindex))*adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space
        print('Env-Info: Hopper Torso-Thigh')

    def _adv_to_xfrc(self, adv_act):
        new_xfrc = self.model.data.xfrc_applied*0.0
        for i,bindex in enumerate(self._adv_bindex):
            new_xfrc[bindex] = np.array([adv_act[i*2], 0., adv_act[i*2+1], 0., 0., 0.])
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

        posbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        posafter,height,ang = self.model.data.qpos[0:3,0]

        # original
        # alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()


        # zp gai

        v = (posafter - posbefore) / self.dt
        # print(v)
        reward_run = np.square(v - self.des_v)
        # reward_run = -0.7 * 1/(np.abs(v - self.des_v)+0.1)
        # print(reward_run)
        # print(v - self.des_v)
        reward_ctrl = 1e-3 * np.square(a).sum()
        reward = reward_run + reward_ctrl - self.alive_bonus

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .3))
        # if done:
        #     print('state', (np.abs(s[2:]) < 100).all(), 'height', (height > .7), 'angle',(abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat,-10,10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_model_zero(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
