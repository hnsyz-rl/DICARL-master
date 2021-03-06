import numpy as np
from gym import utils,spaces
from gym.envs.adversarial.mujoco import mujoco_env

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self.des_v = 3.5
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        ## Adversarial setup
        self._adv_f_bname = b'torso' #Byte String name of body on which the adversary force will be applied
        bnames = self.model.body_names
        self._adv_bindex = bnames.index(self._adv_f_bname) #Index of the body on which the adversary force will be applied
        adv_max_force = 5.
        high_adv = np.ones(2)*adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space

    def _adv_to_xfrc(self, adv_act):
        new_xfrc = self.model.data.xfrc_applied*0.0
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
        posbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        posafter,height,ang = self.model.data.qpos[0:3,0]
        # alive_bonus = 1.0
        # reward = ((posafter - posbefore) / self.dt )
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()

        alive_bonus = 8.0
        v = (posafter - posbefore) / self.dt
        # reward_run = - 1.5 * v
        reward_run = -0.5 * 1/(np.square(v - self.des_v)+0.1)

        reward_ctrl = -1e-3 * (0.5 * 1/np.square(a).sum())
        # reward_ctrl = 0
        reward = reward_run + reward_ctrl - alive_bonus


        done = not (height > 1.0 and height < 2.0
                    and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel,-10,10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def reset_model_zero(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
