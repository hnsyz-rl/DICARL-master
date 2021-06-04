"""
Training script for RARL framework
"""

import argparse
import logging
import time


import DICARL

from gymfc_nf.envs import *
import gym
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import pposgd_simple

import numpy as np

import MlpPolicy_Pro
import MlpPolicy_Adv
from baselines.ppo1.mlp_policy import MlpPolicy

import tensorflow as tf
import os
import logger as lg

class StepCallback:

    def __init__(self, total_timesteps, log_freq=1):
        """
        Args:
            total_timesteps: Total timesteps for training
            log_freq: Number of episodes until an update log message is printed
        """
        self.timesteps = total_timesteps
        self.steps_taken = 0
        self.es = []
        self.sps = []
        self.ep = 1
        self.rewards = []
        self.log_freq = log_freq
        self.log_header = ["Ep",
                           "Done",
                           "Steps",

                           "r",
                           "-ydelta",
                           "+ymin",
                           "+/-e",
                           "-ahigh",
                           "-nothing",

                           "score",

                           "pMAE",
                           "qMAE",
                           "rMAE"]

        header_format = ["{:<5}",
                         "{:<7}",
                         "{:<15}",

                         "{:<15}",
                         "{:<15}",
                         "{:<15}",
                         "{:<15}",
                         "{:<15}",
                         "{:<15}",

                         "{:<10}",

                         "{:<7}",
                         "{:<7}",
                         "{:<7}"]
        self.header_format = "".join(header_format)

        log_format_entries = ["{:<5}",
                              "{:<7.0%}",
                              "{:<15}",

                              "{:<15.0f}",
                              "{:<15.0f}",
                              "{:<15.0f}",
                              "{:<15.0f}",
                              "{:<15.0f}",
                              "{:<15.0f}",

                              "{:<10.2f}",

                              "{:<7.0f}",
                              "{:<7.0f}",
                              "{:<7.0f}"]

        self.log_format = "".join(log_format_entries)

    def callback(self, local, state, reward, done):

        self.es.append(local.true_error)
        self.sps.append(local.angular_rate_sp)

        assert local.ind_rewards[0] <= 0 # oscillation penalty
        assert local.ind_rewards[1] >= 0 # min output reward
        assert local.ind_rewards[3] <= 0 # over saturation penalty
        assert local.ind_rewards[4] <= 0 # do nothing penalty

        self.rewards.append(local.ind_rewards)

        if done:
            if self.ep == 1:
                print(self.header_format.format(*self.log_header))
            # XXX (wfk) Try this new score, we need something normalized to handle the
            # random setpoints. Scale by the setpoint, larger setpoints incur
            # more error. +1 prevents divide by zero
            mae = np.mean(np.abs(self.es))
            mae_pqr = np.mean(np.abs(self.es), axis=0)
            e_score = mae / (1 + np.mean(np.abs(self.sps)))
            self.steps_taken += local.step_counter

            if self.ep % self.log_freq == 0:
                ave_ind_rewards = np.mean(self.rewards, axis=0)
                ind_rewards = ""
                for r in ave_ind_rewards:
                    ind_rewards += "{:<15.2f} ".format(r)

                log_data = [
                    self.ep,
                    self.steps_taken/self.timesteps,
                    self.steps_taken,

                    np.mean(self.rewards),
                    ave_ind_rewards[0],
                    ave_ind_rewards[1],
                    ave_ind_rewards[2],
                    ave_ind_rewards[3],
                    ave_ind_rewards[4],

                    e_score,
                    mae_pqr[0],
                    mae_pqr[1],
                    mae_pqr[2]
                ]
                print (self.log_format.format(*log_data))

            self.ep += 1
            self.es = []
            self.sps = []
            self.rewards = []




def train_DICARL(env_id, num_timesteps, seed, render, max_steps_episode, gymfc_model_path, clip_action=False, ckpt_dir=None, restore_dir=None, n=1.0):

    def policy_pro(name, ob_space, ac_space):
        # return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #                            hid_size=64, num_hid_layers=2)
        return MlpPolicy_Pro.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, tau=3e-4,
                                   hid_size=32, num_hid_layers=2)  # 倒立双摆 4层

    def policy_adv(name, ob_space, ac_space):
        # return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #                            hid_size=64, num_hid_layers=2)
        return MlpPolicy_Adv.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, tau=3e-4,
                                   hid_size=32, num_hid_layers=2)  # 倒立双摆 4层

    env = gym.make(env_id)
    def sample_noise(inst):
        # Experiementally derived for MatekF7 FC, see Chapter 5 of "Flight
        # Controller Synthesis Via Deep Reinforcement Learning" for methodology.
        r_noise = inst.np_random.normal(-0.25465, 1.3373)
        p_noise = inst.np_random.normal(0.241961, 0.9990)
        y_noise = inst.np_random.normal(0.07906, 1.45168)
        return np.array([r_noise, p_noise, y_noise])

    env.sample_noise = sample_noise
    env.set_aircraft_model(gymfc_model_path)

    cb = StepCallback(num_timesteps)
    env.step_callback = cb.callback

    env.update_adversary(1.0)
    # rank = MPI.COMM_WORLD.Get_rank()
    # workerseed = seed + 1000000 * rank
    env.seed(seed)
    set_global_seeds(seed)

    if render:
        env.render()
    gym.logger.setLevel(logging.WARN)
    save_timestep_period = num_timesteps/100
    pro_pi, rew, timesteps_so_far, len_mean= DICARL.learn(env, policy_pro, policy_adv,
                         max_timesteps=num_timesteps,
                         timesteps_per_batch=512,
                         clip_param=0.2, entcoeff=0.0,
                         optim_epochs=5, optim_stepsize=1e-4, optim_batchsize=64, max_steps_episode=max_steps_episode,
                         gamma=0.99, lam=0.95, lr_l=1e-4, lr_a=1e-4, schedule='linear', clip_action=clip_action, alpha=n,
                         restore_dir=restore_dir, 
                         ckpt_dir=ckpt_dir,
                         save_timestep_period=save_timestep_period,
                         )

    env.close()
    return pro_pi, len_mean, timesteps_so_far



def impulse_evaluation(env, ob, action, t, impulse_add_adv_time, magnitude):
    ac = env.sample_action()  # not used, just so we have the datatype
    ac.pro = action

    if t == impulse_add_adv_time:
        # print('加入干扰啦')
        d_init = np.ones(env.adv_action_space.shape[0])
        d = d_init * magnitude * np.sign(ob[0])
    else:
        d = np.zeros(env.adv_action_space.shape[0])
    ac.adv = d
    ob, r, done, _ = env.step(ac)
    return ob, r, done


def continue_impulse(env, ob, action, t, each_impulse_add_adv_time, magnitude):
    ac = env.sample_action()  # not used, just so we have the datatype
    ac.pro = action
    d_dim = env.adv_action_space.shape[0]
    if (t + 1) % each_impulse_add_adv_time == 0:
        d_init = np.ones(d_dim)
        if 'Pendulum' in env.spec.id:
            d = d_init * magnitude * np.sign(ob[0])
        elif 'HalfCheetahTorsoAdv' in env.spec.id:
            d = d_init * magnitude * np.sign(np.random.uniform(-1, 1, d_dim))
        elif 'HumanoidHeelAdv' in env.spec.id:
            d = d_init * magnitude * np.sign(np.random.uniform(-1, 1, d_dim))
        else:
            d = d_init * magnitude * np.sign(ob[0])

    else:
        d = np.zeros(env.adv_action_space.shape[0])
    ac.adv = d
    ob, r, done, _ = env.step(ac)
    return ob, r, done


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env', help='environment ID', default='HalfCheetahAdv-v1')

    # parser.add_argument('--env', help='environment ID', default='HalfCheetahHeelAdv-v1')

    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--adversary', help='Numbers of adversary forces', type=int, default=1)

    parser.add_argument('--num_timesteps', help='Numbers of timesteps', type=int, default=1e7)

    model_path_pre = '../log/'

    gymfc_model_path = './gymfc_nf/twins/nf1/model.sdf'
    env_id = 'gymfc_nf-step-v1'
    # [b'torso', b'lwaist', b'pelvis', b'right_thigh', b'right_shin', b'right_foot', b'left_thigh', b'left_shin', b'left_foot', b'right_upper_arm', b'right_lower_arm', b'left_upper_arm', b'left_lower_arm']

    seed_buffer_zp = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 67, 71, 73, 79, 81]

    parser.add_argument('--env', help='environment ID', default=env_id)
    args = parser.parse_args()

    date = '0602'
    adv_mag = 0.03
    training_name = 'DICARL-1e7-' + date + '-Lclip_003-MD_' + str(adv_mag).replace('.', '_') + '-rew_1e6-rate_100'





    is_train = True
    clip_action = True
    Render = False

    max_step = 4096


    this_start_of_trial = 0
    this_end_of_trial = 1

    if is_train:
        # train the model

        Timesteps_buffer = np.zeros((this_end_of_trial - this_start_of_trial))
        Len_buffer = np.zeros((this_end_of_trial - this_start_of_trial))
        j = 0
        for i in range(this_start_of_trial, this_end_of_trial):
            tf.reset_default_graph()
            sess = U.make_session(num_cpu=1)
            sess.__enter__()

            print('第%d次：' % i)
            model_path = model_path_pre + env_id + '/' + date + '/' + training_name + '/' + str(
                i) + '/'  # 这里是生成一个日志存放的路径：~/log/环境名/算法名+算法描述名/迭代次数i
            os.makedirs(model_path, exist_ok=True)
            print('logging to ' + model_path)  # 这两行类似于日志，但我不太懂为啥循环从i=5开始(start_of_trial=5)
            # seed = int(time.time() * 1e7 % 61)
            # seed = 0
            seed = seed_buffer_zp[i]


            _, len_mean, stepsfar = train_DICARL(env_id=env_id, num_timesteps=args.num_timesteps, seed=seed, render=Render,
                                                  clip_action=clip_action, ckpt_dir=model_path, restore_dir=None, n=adv_mag,
                                                max_steps_episode= max_step, gymfc_model_path=gymfc_model_path)
            print(training_name)
            Timesteps_buffer[j] = stepsfar
            Len_buffer[j] = len_mean
            j+=1
            sess.__exit__(None, None, None)

        print(Timesteps_buffer)
        print(Len_buffer)




