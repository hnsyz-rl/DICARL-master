import numpy as np
from gym import utils, spaces
from gym.envs.adversarial.mujoco import mujoco_env

class HalfCheetahTorsoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.alive_bonus = 20.

        self.des_v = 3.5
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        ## Adversarial setup
        self._adv_f_bname = [b'torso', b'bfoot', b'ffoot'] #Byte String name of body on which the adversary force will be applied
        bnames = self.model.body_names
        self._adv_bindex = [bnames.index(i) for i in self._adv_f_bname] #Index of the body on which the adversary force will be applied
        adv_max_force = 5.
        high_adv = np.ones(2*len(self._adv_bindex))*adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space


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

        xposbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.model.data.qpos[0,0]
        ob = self._get_obs()
        reward_ctrl = 0.05 * np.square(a).sum()
        # reward_run = (xposafter - xposbefore)/self.dt
        v = (xposafter - xposbefore) / self.dt
        # print(v)
        # if np.abs(v) > 2:
        #     print('v', v)
        reward_run = np.square(v - self.des_v)
        reward = (reward_ctrl + reward_run - self.alive_bonus)
        # reward = (reward_ctrl + reward_run) * 0.5

        # print('reward_ctrl', reward_ctrl)
        # print('reward_run', reward_run)
        # print('reward', reward)



        if abs(ob[1]) > 1.0: # np.pi/2:  # ??????ob1?????????????????????
            done = True
        else:
            done = False

        return ob, reward, done, dict(reward_run = reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
