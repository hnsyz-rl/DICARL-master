import numpy as np
from gym.envs.adversarial.mujoco import mujoco_env
from gym import utils,spaces

def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidHeelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.alive_bonus = 5.0
        self.des_v = 3.5

        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)
        ## Adversarial setup
        self._adv_f_bname = [b'torso', b'lwaist'] #Byte String name of body on which the adversary force will be applied

# [b'world', b'torso', b'lwaist', b'pelvis', b'right_thigh', b'right_shin', b'right_foot', b'left_thigh', b'left_shin', b'left_foot', b'right_upper_arm', b'right_lower_arm', b'left_upper_arm', b'left_lower_arm']


        bnames = self.model.body_names
        self._adv_bindex = [bnames.index(i) for i in self._adv_f_bname]#Index of the body on which the adversary force will be applied
        adv_max_force = 10.0
        high_adv = np.ones(3*len(self._adv_bindex))*adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space
        print('\n\nEnv InFo: Humanoid_hell, adv: torso, pelvis\n\n')
    def _adv_to_xfrc(self, adv_act):
        new_xfrc = self.model.data.xfrc_applied*0.0
        for i, bindex in enumerate(self._adv_bindex):
            new_xfrc[bindex] = np.array([adv_act[i * 3], adv_act[i * 3 + 1], adv_act[i * 3 + 2], 0., 0., 0.])
        self.model.data.xfrc_applied = new_xfrc

    def sample_action(self):
        class act(object):
            def __init__(self,pro=None,adv=None):
                self.pro=pro
                self.adv=adv
        sa = act(self.pro_action_space.sample(), self.adv_action_space.sample())
        return sa

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def _step(self, action):
        if hasattr(action, '__dict__'):
            self._adv_to_xfrc(action.adv)
            a = action.pro
        else:
            a = action

        pos_before = mass_center(self.model)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model)
        data = self.model.data
        lin_vel_cost = 0.75 * (pos_after - pos_before) / self.model.opt.timestep
        # ?????????????????????0.5??????????????????????????????????????????????????????????????????
        # zp
        v = (pos_after - pos_before) / self.dt
        reward_run = np.square(v - self.des_v)

        #zp

        quad_ctrl_cost =  0.01 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-7 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + self.alive_bonus
        # reward = reward_run + quad_ctrl_cost + quad_impact_cost - self.alive_bonus

        qpos = self.model.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        # return self._get_obs(), -reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=self.alive_bonus, reward_impact=-quad_impact_cost)
        return self._get_obs(), -reward * 0.1, done, v

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def reset_model_zero(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
