"""
Training script for RARL framework
"""

import argparse
import logging
import time

import RARL_PPO
import DICARL
import Domain_rand
import NR_MDP_PPO
import Vanllia_PPO


import gym
from baselines.common import set_global_seeds, tf_util as U

import numpy as np

import MlpPolicy_Pro
import MlpPolicy_Adv

import tensorflow as tf
import os
import logger as lg


def train_RARL(env_id, num_iters, num_timesteps, seed, model_path, max_steps_episode, n=1.0, clip_action=True):


    def policy_pro(name, ob_space, ac_space):
        return MlpPolicy_Pro.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, tau=3e-4,
                                   hid_size=64, num_hid_layers=4)  # 倒立双摆 4层

    def policy_adv(name, ob_space, ac_space):
        return MlpPolicy_Adv.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, tau=3e-4,
                                   hid_size=64, num_hid_layers=4)  # 倒立双摆 4层

    env = gym.make(env_id)

    env.update_adversary(n)
    set_global_seeds(seed)

    env.seed(seed)

    gym.logger.setLevel(logging.WARN)

    pro_pi, rew, len_mean, stepsfar = RARL_PPO.learn(env, policy_pro, policy_adv,
                                 max_timesteps=num_timesteps,
                                 timesteps_per_batch=2048,
                                 clip_param=0.02, entcoeff=0.0,
                                 optim_epochs=10, optim_stepsize=5e-4, optim_batchsize=64, max_steps_episode=max_steps_episode,
                                 gamma=0.99, lam=0.95, schedule='linear', clip_action= clip_action
                                 )  # 这个超参数好，optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,

    env.close()

    if model_path:
        # print(model_path)
        U.save_state(model_path)

    return pro_pi, len_mean, stepsfar


def train_PPO(env_id, num_iters, num_timesteps, seed, model_path, max_steps_episode, n=1.0, clip_action=True):


    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy_Pro.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, tau=3e-4,
                                       hid_size=64, num_hid_layers=4)  # 倒立双摆 4层
    set_global_seeds(seed)

    env = gym.make(env_id)
    env.update_adversary(n)

    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    pi, rew, len_mean, stepsfar = Vanllia_PPO.learn(env, policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=2048,
                        clip_param=0.02, entcoeff=0.0,
                        optim_epochs=10,
                        optim_stepsize=5e-4,
                        optim_batchsize=64, max_steps_episode=max_steps_episode,
                        gamma=0.99, lam=0.95, schedule='linear',
                        )
    env.close()

    if model_path:
        # print(model_path)
        U.save_state(model_path)

    return pi, len_mean, stepsfar

def train_NR_MDP_PPO(env_id, num_iters, num_timesteps, seed, model_path, max_steps_episode, alpha, n=1.0, clip_action=True):


    def policy_pro(name, ob_space, ac_space):
        return MlpPolicy_Pro.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, tau=3e-4,
                                   hid_size=64, num_hid_layers=4)  # 倒立双摆 4层

    def policy_adv(name, ob_space, ac_space):
        return MlpPolicy_Adv.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, tau=3e-4,
                                   hid_size=64, num_hid_layers=4)  # 倒立双摆 4层

    env = gym.make(env_id)

    env.update_adversary(n)
    set_global_seeds(seed)

    env.seed(seed)

    gym.logger.setLevel(logging.WARN)

    pro_pi, rew, len_mean, stepsfar = NR_MDP_PPO.learn(env, policy_pro, policy_adv,
                                 max_timesteps=num_timesteps, alpha=alpha,
                                 timesteps_per_batch=2048,
                                 clip_param=0.02, entcoeff=0.0,
                                 optim_epochs=10, optim_stepsize=5e-4, optim_batchsize=64, max_steps_episode=max_steps_episode,
                                 gamma=0.99, lam=0.95, schedule='linear', clip_action= clip_action
                                 )  # 这个超参数好，optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,

    env.close()

    if model_path:
        # print(model_path)
        U.save_state(model_path)

    return pro_pi, len_mean, stepsfar


def train_Domain_rand(env_id, num_iters, num_timesteps, seed, model_path, param_variation, max_steps_episode, n=1.0, clip_action=True):


    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy_Pro.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, tau=3e-4,
                                       hid_size=64, num_hid_layers=4)  # 倒立双摆 4层
    set_global_seeds(seed)

    env = gym.make(env_id)
    env.update_adversary(n)

    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    pi, rew, len_mean, stepsfar = Domain_rand.learn(env, policy_fn,
                        param_variation=param_variation,
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=2048,
                        clip_param=0.02, entcoeff=0.0,
                        optim_epochs=10,
                        optim_stepsize=5e-4,
                        optim_batchsize=64, max_steps_episode=max_steps_episode,
                        gamma=0.99, lam=0.95, schedule='linear',
                        )
    env.close()

    if model_path:
        # print(model_path)
        U.save_state(model_path)

    return pi, len_mean, stepsfar


def train_DICARL(env_id, num_timesteps, seed, render, max_steps_episode, clip_action=False, ckpt_dir=None, restore_dir=None, n=1.0):

    def policy_pro(name, ob_space, ac_space):
        # return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #                            hid_size=64, num_hid_layers=2)
        return MlpPolicy_Pro.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, tau=3e-4,
                                   hid_size=64, num_hid_layers=4)  # 倒立双摆 4层

    def policy_adv(name, ob_space, ac_space):
        # return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        #                            hid_size=64, num_hid_layers=2)
        return MlpPolicy_Adv.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, tau=3e-4,
                                   hid_size=64, num_hid_layers=4)  # 倒立双摆 4层

    env = gym.make(env_id)


    env.update_adversary(n)

    set_global_seeds(seed)
    env.seed(seed)

    save_timestep_period = num_timesteps
    if ckpt_dir:
        print('logging to ' + ckpt_dir)
    pro_pi, rew, timesteps_so_far, len_mean= DICARL.learn(env, policy_pro, policy_adv,
                         max_timesteps=num_timesteps,
                         timesteps_per_batch=2048,
                         clip_param=0.02, entcoeff=0.0,
                         optim_epochs=10, optim_stepsize=5e-4, optim_batchsize=64, max_steps_episode=max_steps_episode,
                         gamma=0.99, lam=0.95, lr_l=5e-4, lr_a=5e-4, schedule='linear', clip_action=clip_action,
                         restore_dir=restore_dir,
                         ckpt_dir=None,
                         save_timestep_period=save_timestep_period,
                         )
    if ckpt_dir:
        # print(model_path)
        U.save_state(ckpt_dir)

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
        elif 'Hopper' in env.spec.id:
            d = d_init * magnitude * np.sign(np.random.uniform(-1, 1, d_dim))
            d[2] = 0
            d[3] = 0
        else:
            d = d_init * magnitude * np.sign(ob[0])

    else:
        d = np.zeros(env.adv_action_space.shape[0])
    ac.adv = d
    ob, r, done, _ = env.step(ac)
    return ob, r, done


if __name__ == '__main__':


    model_path_pre = '../log/'
    num_timesteps = 2e6
    adv_mag = 3.8
    env_id = 'HopperHeelAdv-v1'


    training_name = 'RARL-2e6-action-clip-0601-leg-adv_1_3'  # 在优势函数里去了

    # test_name = 'DICARL-2e6-baselines-action-clip-0328_Lclip_003_linear-d-3_8'
    # test_name = 'Domain_rand-2e6-baselines-action-clip-deep_4-0320'
    # test_name = 'NR_MDP-2e6-baselines-action-clip-deep_4-0427-alpha01'
    # test_name = 'PPO-2e6-baselines-action-clip-0427'
    test_name = 'RARL-2e6-action-clip-0322-leg-adv_5_3'



    is_train = False
    clip_action = True
    Render = False


    # evaluation_case = "impulse_response"
    # evaluation_case = 'continue_impulse'
    evaluation_case = 'param_variation'

    # adversary setting


    impulse_add_adv_time = 250
    each_impulse_add_adv_time = 10
    impulse_min_magnitude = 0
    impulse_max_magnitude = 150
    impulse_step_magnitude = 10
    eval_num_total = 70
    test_num = 10
    max_step = 2000


    param_variation = {
        'param_variables': {
            'mass_of_torso': np.arange(0.1, 2.1, 0.2),  # 0.1 # 躯干质量
            'mass_of_foot': np.arange(0.1, 2.1, 0.2),  # 1.0 # 足部质量
            'mass_of_thigh': np.arange(0.1, 2.1, 0.2),  # 1.0 # 足部质量
            'mass_of_leg': np.arange(0.1, 2.1, 0.2),  # 1.0 # 足部质量

            'friction': np.arange(0.1, 2.1, 0.2)
            # 'gravity': np.arange(9, 10.1, 0.1),  # 0.1 # 哎呦呵，还有重力变化，不过这里没用
        },
        'grid_eval': True,  # 不知道干嘛的，可能是开关网格？不对，这个因该是一个网格采样，就是生成一个由上面序列组合成的矩阵
        # 'grid_eval': False,
        'friction_eval': False,
        'impulse_flag': False,
        'friction_name': [b'foot'],

        # 'grid_eval_param_name': [b'torso', b'foot'],
        # 'grid_eval_param': ['mass_of_torso', 'mass_of_foot'],

        'grid_eval_param_name': [b'torso', b'leg'],
        'grid_eval_param': ['mass_of_torso', 'mass_of_leg'],

        # 'grid_eval_param_name': [b'torso', b'thigh'],
        # 'grid_eval_param': ['mass_of_torso', 'mass_of_thigh'],
        
        # 'grid_eval_param_name': [b'leg', b'foot'],
        # 'grid_eval_param': ['mass_of_leg', 'mass_of_foot'],
        
        # 'grid_eval_param_name': [b'thigh', b'foot'],
        # 'grid_eval_param': ['mass_of_thigh', 'mass_of_foot'],
        
        # 'grid_eval_param_name': [b'leg', b'thigh'],
        # 'grid_eval_param': ['mass_of_leg', 'mass_of_thigh'],

        # 'grid_eval_param': ['mass_of_torso', 'friction'],

        'impulse_instant': 200,  # 加入冲击的时刻
    }

    log_case = 'death_rate'
    # log_case = 'cost'
    # log_case = 'len'





    this_start_of_trial = 0
    this_end_of_trial = 7
    seed_buffer_zp = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 67, 71, 73, 79, 81]

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
            model_path = model_path_pre + env_id + '/' + training_name + '/' + str(
                i) + '/'  # 这里是生成一个日志存放的路径：~/log/环境名/算法名+算法描述名/迭代次数i
            os.makedirs(model_path, exist_ok=True)
            print('logging to ' + model_path)  # 这两行类似于日志，但我不太懂为啥循环从i=5开始(start_of_trial=5)
            # seed = int(time.time() * 1e7 % 61)
            # seed = 0
            seed = seed_buffer_zp[i]

            if 'PPO' in training_name:
                _, len_mean, stepsfar  = train_PPO(num_iters=1e6, num_timesteps=num_timesteps, seed=seed, model_path=model_path,
                      env_id=env_id, n=adv_mag, clip_action=clip_action, max_steps_episode=max_step)
            elif 'RARL' in training_name:
                _, len_mean, stepsfar = train_RARL(num_iters=1e6, num_timesteps=num_timesteps, seed=seed, model_path=model_path,
                      env_id=env_id, n=adv_mag, clip_action=clip_action, max_steps_episode=max_step)
            elif 'NR_MDP' in training_name:
                _, len_mean, stepsfar = train_NR_MDP_PPO(num_iters=1e6, num_timesteps=num_timesteps, seed=seed, model_path=model_path,
                      env_id=env_id, n=0, alpha=0.1, clip_action=clip_action, max_steps_episode=max_step)
            elif 'Domain' in training_name:
                _, len_mean, stepsfar = train_Domain_rand(num_iters=1e6, num_timesteps=num_timesteps, seed=seed, model_path=model_path,
                      env_id=env_id, n=adv_mag, param_variation=param_variation, clip_action=clip_action, max_steps_episode=max_step)
            else:
                _, len_mean, stepsfar = train_DICARL(env_id=env_id, num_timesteps=num_timesteps, seed=seed, render=Render,
                                                      clip_action=clip_action, ckpt_dir=model_path, restore_dir=None, n=adv_mag,
                                                    max_steps_episode= max_step)
            print(training_name)
            Timesteps_buffer[j] = stepsfar
            Len_buffer[j] = len_mean
            j+=1
            sess.__exit__(None, None, None)

        print(Timesteps_buffer)
        print(Len_buffer)





    if ((evaluation_case == 'param_variation' and is_train) or not is_train):
        if is_train:
            test_name = training_name
        print(" Making env=", env_id)
        # construct the model object, load pre-trained model and render
        test_dir = model_path_pre + env_id + '/' + test_name + '/'

        trial_list = os.listdir(test_dir)
        print(trial_list)
        magnitude_list_len = len(range(impulse_min_magnitude, impulse_max_magnitude, impulse_step_magnitude))
        if 'eval' in trial_list:
            eval_list_len = len(trial_list) - 1
        else:
            eval_list_len = len(trial_list)
        eval_num = int(np.ceil(eval_num_total / eval_list_len))
        print('eval_num', eval_num)

        death_rates = np.zeros((eval_list_len, magnitude_list_len))
        returns = np.zeros((eval_list_len, magnitude_list_len))
        # 参数变化测试相关参数
        param_variable = param_variation['param_variables']  # 获取参数变化
        grid_eval_param = param_variation['grid_eval_param']  # 获取网格采样
        param1 = grid_eval_param[0]  # 获取参数1，源码里定义的是车杆长度
        param2 = grid_eval_param[1]  # 获取参数2，源码里定义的是车子质量
        param1_len = len(param_variable[param1])
        param2_len = len(param_variable[param2])

        log_dir = test_dir + 'eval' + '/' + evaluation_case + '/'
        if evaluation_case == 'param_variation':
            log_dir = log_dir + param1 + '-' + param2 + '/'
        lg.configure(dir=log_dir, format_strs=['csv'])

        if 'impulse' in evaluation_case:
            test_num = 1

        final_death_rate = np.zeros((param1_len, param2_len, test_num))
        final_cost = np.zeros((param1_len, param2_len, test_num))
        final_len = np.zeros((param1_len, param2_len, test_num))

        for num in range(test_num):

            seed = seed_buffer_zp[num]

            all_death_rate = np.zeros((param1_len, param2_len, eval_list_len))
            all_cost = np.zeros((param1_len, param2_len, eval_list_len))
            all_len = np.zeros((param1_len, param2_len, eval_list_len))
            n = 0

            for i in trial_list:

                if i == 'eval':
                    continue

                tf.reset_default_graph()

                sess = U.make_session(num_cpu=1)
                sess.__enter__()
                if 'PPO' in test_name:
                    pi, _, _ = train_PPO(num_iters=1, num_timesteps=1, seed=seed, env_id=env_id, model_path=False,
                                         max_steps_episode=max_step)

                elif 'RARL' in test_name:
                    pi, _, _ = train_RARL(num_iters=1, num_timesteps=1, seed=seed, env_id=env_id, model_path=False,
                                          max_steps_episode=max_step)
                elif 'NR_MDP' in test_name:
                    pi, _, _ = train_NR_MDP_PPO(num_iters=1, num_timesteps=1, seed=seed, env_id=env_id,
                                                model_path=False,
                                                n=0, alpha=0.1, max_steps_episode=max_step)
                elif 'Domain' in test_name:
                    pi, _, _ = train_Domain_rand(num_iters=1, num_timesteps=1, seed=seed, model_path=False,
                                                 env_id=env_id, n=adv_mag, param_variation=param_variation,
                                                 max_steps_episode=max_step)
                else:
                    pi, _, _ = train_DICARL(env_id=env_id, num_timesteps=1, seed=seed, render=Render, ckpt_dir=False,
                                             restore_dir=None, n=adv_mag, max_steps_episode=max_step)

                model_path = test_dir + str(i) + '/'  # 这里是生成一个日志存放的路径：~/log/环境名/算法名+算法描述名/迭代次数i
                print(model_path)
                U.load_state(model_path)
                print('\ntotal num', num, 'sub num', n,'\n')
                env = gym.make(env_id)
                a_upperbound = env.action_space.high
                a_lowerbound = env.action_space.low
                die_num = 0

                per_mag_cost = []
                death_rates_each_model = []
                per_mag_len = []

                m = 0

                if 'impulse' in evaluation_case:
                    for magnitude in range(impulse_min_magnitude, impulse_max_magnitude, impulse_step_magnitude):

                        each_cost = []
                        each_len = []

                        for i_episode in range(eval_num):

                            ob = env.reset()
                            t = 0
                            cost = 0
                            while True:

                                ac, _ = pi.act(stochastic=False, ob=ob)

                                if clip_action:
                                    action = a_lowerbound + (np.tanh(ac) + 1.) * (a_upperbound - a_lowerbound) / 2
                                else:
                                    action = ac
                                # print(action)

                                # print('action', action)

                                if Render:
                                    env.render()
                                    if env_id == 'HopperAdv-v1':
                                        time.sleep(0.0025)
                                    else:
                                        time.sleep(0.03)
                                # print('ob:',ob)

                                # for j in range(len(ob)):
                                #     lg.logkv('ob%d' % j, ob[j])
                                # lg.dumpkvs()

                                if evaluation_case == "impulse_response":
                                    ob, r, done = impulse_evaluation(env, ob, action, t, impulse_add_adv_time,
                                                                     magnitude)
                                elif evaluation_case == "continue_impulse":
                                    ob, r, done = continue_impulse(env, ob, action, t, each_impulse_add_adv_time,
                                                                   magnitude)
                                else:
                                    ac = env.sample_action()  # not used, just so we have the datatype
                                    ac.pro = action
                                    d_init = np.ones(env.adv_action_space.shape[0])
                                    d = d_init * 10 * np.sign(ob[0])
                                    ac.adv = d
                                    ob, r, done, _ = env.step(ac)

                                t += 1
                                cost += r

                                if t == max_step:
                                    done = True
                                if done:
                                    # print("Episode finished after {} timesteps".format(t + 1))
                                    if t < max_step:
                                        die_num += 1
                                    break
                            each_len.append(t)
                            each_cost.append(cost)
                        per_mag_cost.append(np.mean(each_cost))
                        per_mag_len.append(np.mean(each_len))

                        death_rate = die_num / eval_num * 100
                        die_num = 0
                        death_rates_each_model.append(death_rate)
                        print('death_rate at %d' % magnitude, ': ', death_rate)
                        print('cost at %d' % magnitude, ': ', np.mean(each_cost))

                        death_rates[n][m] = death_rate
                        returns[n][m] = np.mean(each_cost)
                        # lg.logkv('model_%s_death' % i, death_rate)
                        # lg.logkv('model_%s_cost' % i, np.mean(each_cost))
                        # lg.dumpkvs()
                        m += 1

                    print('cost', per_mag_cost)
                    print('len', per_mag_len)
                    print('death_rate', death_rates_each_model)
                elif 'param' in evaluation_case:

                    # length_of_pole, mass_of_pole, mass_of_cart, gravity = env.get_params()  # 获取环境变量中杆子长度、杆子质量、车质量、重力

                    adv_f_bname = param_variation[
                        'grid_eval_param_name']  # Byte String name of body on which the adversary force will be applied
                    friction_bname = param_variation['friction_name']
                    bnames = env.model.body_names  # 这里就是从mujoco中获取环境中包含的信息
                    adv_bindex = [bnames.index(i) for i in
                                  adv_f_bname]  # Index of the body on which the adversary force will be applied

                    friction_bindex = [bnames.index(i) for i in friction_bname]

                    model_mass = env.model.body_mass
                    model_friction = env.model.geom_friction

                    zp_i = 0
                    for var1 in param_variable[param1]:
                        zp_j = 0
                        for var2 in param_variable[param2]:
                            if param_variation['friction_eval']:
                                model_mass_variable = np.array(model_mass)
                                model_mass_variable[adv_bindex[0], 0] = model_mass_variable[adv_bindex[0], 0] * var1
                                model_friction_variable = np.array(model_friction)
                                findex = friction_bindex
                                model_friction_variable[findex, 0] = model_friction_variable[findex, 0] * var2
                                env.model.geom_friction = model_friction_variable

                            else:
                                model_mass_variable = np.array(model_mass)
                                model_mass_variable[adv_bindex[0], 0] = model_mass_variable[adv_bindex[0], 0] * var1
                                model_mass_variable[adv_bindex[1], 0] = model_mass_variable[adv_bindex[1], 0] * var2

                            env.model.body_mass = model_mass_variable

                            each_cost = []
                            each_len = []

                            for i_episode in range(eval_num):
                                ob = env.reset()
                                t = 0
                                cost = 0
                                while True:

                                    ac, _ = pi.act(stochastic=False, ob=ob)

                                    if clip_action:
                                        action = a_lowerbound + (np.tanh(ac) + 1.) * (a_upperbound - a_lowerbound) / 2
                                    else:
                                        action = ac

                                    if Render:
                                        env.render()
                                        time.sleep(0.03)
                                    magnitude = 40
                                    if param_variation['impulse_flag']:
                                        ob, r, done = impulse_evaluation(env, ob, action, t, impulse_add_adv_time,
                                                                         magnitude)
                                    else:
                                        ob, r, done, _ = env.step(action)

                                    t += 1
                                    cost += r

                                    if t == max_step:
                                        done = True
                                    if done:
                                        # print("Episode finished after {} timesteps".format(t + 1))
                                        if t < max_step:
                                            die_num += 1
                                        break
                                each_len.append(t)
                                each_cost.append(cost)

                            death_rate = die_num / eval_num * 100

                            all_cost[zp_i][zp_j][n] = np.mean(each_cost)
                            all_len[zp_i][zp_j][n] = np.mean(each_len)
                            all_death_rate[zp_i][zp_j][n] = (death_rate)
                            print('Info at %s:%.2f | %s:%.2f' % (param1, var1, param2, var2),
                                  ': dr: %.1f, cost: %.2f, len: %.1f' %
                                  (death_rate, np.mean(each_cost), np.mean(each_len)))

                            die_num = 0
                            zp_j += 1
                        zp_i += 1

                # sess2 = tf.get_default_session()
                # sess2.close()
                # tf.get_default_graph()
                # sess1.close()
                # death_rates = death_rates + np.array(death_rates_each_model)
                n += 1
                # lg.dumpkvs()

                sess.__exit__(None, None, None)
            for zp_i in range(param1_len):
                for zp_j in range(param2_len):
                    final_death_rate[zp_i][zp_j][num] = np.mean(all_death_rate[zp_i][zp_j][:])
                    final_cost[zp_i][zp_j][num] = np.mean(all_cost[zp_i][zp_j][:])
                    final_len[zp_i][zp_j][num] = np.mean(all_len[zp_i][zp_j][:])

        # death_rates = death_rates / eval_list_len
        # print('death_rates:', death_rates)
        if 'impulse' in evaluation_case:
            magnitude = range(impulse_min_magnitude, impulse_max_magnitude, impulse_step_magnitude)
            death_rates_mean = np.mean(death_rates, axis=0)
            death_rates_std = np.std(death_rates, axis=0)
            print('death_rates_mean')
            for each_death_rate in death_rates_mean:
                print("%.2f" % each_death_rate, end=', ')
            print('\ndeath_rates_std')
            for each_death_rates_std in death_rates_std:
                print("%.2f" % each_death_rates_std, end=', ')
            print('\nend')

            returns_mean = np.mean(returns, axis=0)
            returns_std = np.std(returns, axis=0)
            print('returns_mean')
            for each_return in returns_mean:
                print("%.2f" % each_return, end=', ')
            print('\nreturns_std')
            for each_each_return_std in returns_std:
                print("%.2f" % each_each_return_std, end=', ')
            print('\nend')

            for m in range(magnitude_list_len):
                lg.logkv('magnitude', magnitude[m])
                i = 0
                for n in trial_list:
                    if n == 'eval':
                        continue
                    lg.logkv('model_%s_death' % n, death_rates[i][m])
                    lg.logkv('model_%s_return' % n, returns[i][m])
                    i += 1
                lg.logkv('death_rates_mean', death_rates_mean[m])
                lg.logkv('death_rates_std', death_rates_std[m])
                lg.logkv('returns_mean', returns_mean[m])
                lg.logkv('returns_std', returns_std[m])
                lg.dumpkvs()

        elif 'param' in evaluation_case:
            all_cost_mean = np.zeros((param1_len, param2_len))
            all_len_mean = np.zeros((param1_len, param2_len))
            all_death_rate_mean = np.zeros((param1_len, param2_len))
            # all_cost_std = np.zeros((param1_len, param2_len))
            # all_len_std = np.zeros((param1_len, param2_len))
            # all_death_rate_std = np.zeros((param1_len, param2_len))
            cum_less1_death_rate = 0
            cum_less2_death_rate = 0
            cum_less3_death_rate = 0
            cum_less4_death_rate = 0
            cum_less_total_death_rate = 0

            cum_less1_cost = 0
            cum_less2_cost = 0
            cum_less3_cost = 0
            cum_less4_cost = 0
            cum_less_total_cost = 0

            cum_less1_len = 0
            cum_less2_len = 0
            cum_less3_len = 0
            cum_less4_len = 0
            cum_less_total_len = 0

            cost_max = env.alive_bonus * max_step

            if log_case == 'death_rate':
                print('death_rate:\n[', end='')
            elif log_case == 'cost':
                print('cost:\n[', end='')
            else:
                print('len:\n[', end='')

            for zp_i in range(param1_len):
                lg.logkv('param1_value', param_variable[param1][zp_i])
                for zp_j in range(param2_len):
                    zp_cost = np.mean(final_cost[zp_i][zp_j][:])
                    zp_len = np.mean(final_len[zp_i][zp_j][:])
                    zp_death_rate = np.mean(final_death_rate[zp_i][zp_j][:])

                    all_cost_mean[zp_i][zp_j] = zp_cost
                    all_len_mean[zp_i][zp_j] = zp_len
                    all_death_rate_mean[zp_i][zp_j] = zp_death_rate
                    if log_case == 'death_rate':
                        lg.logkv('param2-%.1f' % param_variable[param2][zp_j], zp_death_rate)
                        print('%-6.1f' % zp_death_rate, end=' ')
                    elif log_case == 'cost':
                        lg.logkv('param2-%.1f' % param_variable[param2][zp_j], zp_cost)
                        print('%-6.2f' % zp_cost, end=' ')
                    else:
                        lg.logkv('param2-%.1f' % param_variable[param2][zp_j], zp_len)
                        print('%-6.1f' % zp_len, end=' ')

                    if zp_death_rate <= 2:
                        cum_less1_death_rate += 1
                    elif 2 < zp_death_rate <= 10:
                        cum_less2_death_rate += 1
                    elif 10 < zp_death_rate <= 30:
                        cum_less3_death_rate += 1
                    elif 30 < zp_death_rate <= 50:
                        cum_less4_death_rate += 1
                    if zp_death_rate <= 50:
                        cum_less_total_death_rate += 1

                    zp_cost = abs(zp_cost)
                    if zp_cost >= cost_max * 0.9:
                        cum_less1_cost += 1
                    elif cost_max * 0.9 > zp_cost >= cost_max * 0.8:
                        cum_less2_cost += 1
                    elif cost_max * 0.8 > zp_cost >= cost_max * 0.7:
                        cum_less3_cost += 1
                    elif cost_max * 0.7 > zp_cost >= cost_max * 0.6:
                        cum_less4_cost += 1
                    if zp_cost >= cost_max * 0.6:
                        cum_less_total_cost += 1

                    if zp_len >= max_step * 0.95:
                        cum_less1_len += 1
                    elif max_step * 0.95 > zp_len >= max_step * 0.8:
                        cum_less2_len += 1
                    elif max_step * 0.8 > zp_len >= max_step * 0.7:
                        cum_less3_len += 1
                    elif max_step * 0.7 > zp_len >= max_step * 0.6:
                        cum_less4_len += 1
                    if zp_len >= max_step * 0.6:
                        cum_less_total_len += 1

                lg.dumpkvs()
                if zp_i == param1_len - 1:
                    print(']\n', end='')
                else:
                    print('\n', end='')

            num_sample = param1_len * param2_len * 0.01
            print('death_rate_analyze: \n<=2 -> %-6.2f\n<=10 -> %-6.2f\n<=30 -> %-6.2f\n<=50 -> %-6.2f' % (
                cum_less1_death_rate / num_sample, cum_less2_death_rate / num_sample, cum_less3_death_rate / num_sample,
                cum_less4_death_rate / num_sample))
            print('total<=50 ->%-6.2f' % (cum_less_total_death_rate / num_sample))

            print('cost_analyze: \n>=0.9 -> %-6.2f\n>=0.8 -> %-6.2f\n>=0.7 -> %-6.2f\n>=0.6 -> %-6.2f' % (
                cum_less1_cost / num_sample, cum_less2_cost / num_sample, cum_less3_cost / num_sample,
                cum_less4_cost / num_sample))
            print('total>=60 ->%-6.2f' % (cum_less_total_cost / num_sample))

            print('len_analyze: \n>=0.95 -> %-6.2f\n>=0.8 -> %-6.2f\n>=0.7 -> %-6.2f\n>=0.6 -> %-6.2f' % (
                cum_less1_len / num_sample, cum_less2_len / num_sample, cum_less3_len / num_sample,
                cum_less4_len / num_sample))
            print('total>=60 ->%-6.2f' % (cum_less_total_len / num_sample))

            print('all_cost_mean\n', all_cost_mean)
            print('all_len_mean\n', all_len_mean)
            print('all_death_rate_mean\n', all_death_rate_mean)

            cum_less1_death_rate = np.zeros(test_num)
            cum_less2_death_rate = np.zeros(test_num)
            cum_less3_death_rate = np.zeros(test_num)
            cum_less4_death_rate = np.zeros(test_num)
            cum_less_total_death_rate = np.zeros(test_num)

            cum_less1_cost = np.zeros(test_num)
            cum_less2_cost = np.zeros(test_num)
            cum_less3_cost = np.zeros(test_num)
            cum_less4_cost = np.zeros(test_num)
            cum_less_total_cost = np.zeros(test_num)

            cum_less1_len = np.zeros(test_num)
            cum_less2_len = np.zeros(test_num)
            cum_less3_len = np.zeros(test_num)
            cum_less4_len = np.zeros(test_num)
            cum_less_total_len = np.zeros(test_num)

            for n in range(test_num):

                for zp_i in range(param1_len):
                    lg.logkv('param1_value', param_variable[param1][zp_i])
                    for zp_j in range(param2_len):
                        zp_cost = (final_cost[zp_i][zp_j][n])
                        zp_len = (final_len[zp_i][zp_j][n])
                        zp_death_rate = (final_death_rate[zp_i][zp_j][n])

                        all_cost_mean[zp_i][zp_j] = zp_cost
                        all_len_mean[zp_i][zp_j] = zp_len
                        all_death_rate_mean[zp_i][zp_j] = zp_death_rate

                        if zp_death_rate <= 2:
                            cum_less1_death_rate[n] += 1
                        elif 2 < zp_death_rate <= 10:
                            cum_less2_death_rate[n] += 1
                        elif 10 < zp_death_rate <= 30:
                            cum_less3_death_rate[n] += 1
                        elif 30 < zp_death_rate <= 50:
                            cum_less4_death_rate[n] += 1
                        if zp_death_rate <= 50:
                            cum_less_total_death_rate[n] += 1

                        zp_cost = abs(zp_cost)
                        if zp_cost >= cost_max * 0.9:
                            cum_less1_cost[n] += 1
                        elif cost_max * 0.9 > zp_cost >= cost_max * 0.8:
                            cum_less2_cost[n] += 1
                        elif cost_max * 0.8 > zp_cost >= cost_max * 0.7:
                            cum_less3_cost[n] += 1
                        elif cost_max * 0.7 > zp_cost >= cost_max * 0.6:
                            cum_less4_cost[n] += 1
                        if zp_cost >= cost_max * 0.6:
                            cum_less_total_cost[n] += 1

                        if zp_len >= max_step * 0.95:
                            cum_less1_len[n] += 1
                        elif max_step * 0.95 > zp_len >= max_step * 0.8:
                            cum_less2_len[n] += 1
                        elif max_step * 0.8 > zp_len >= max_step * 0.7:
                            cum_less3_len[n] += 1
                        elif max_step * 0.7 > zp_len >= max_step * 0.6:
                            cum_less4_len[n] += 1
                        if zp_len >= max_step * 0.6:
                            cum_less_total_len[n] += 1


            print('------------------------------total-------------------------------')
            for zp_i in range(param1_len):
                for zp_j in range(param2_len):
                    zp_cost = np.mean(final_cost[zp_i][zp_j][:])
                    zp_len = np.mean(final_len[zp_i][zp_j][:])
                    zp_death_rate = np.mean(final_death_rate[zp_i][zp_j][:])

                    all_cost_mean[zp_i][zp_j] = zp_cost
                    all_len_mean[zp_i][zp_j] = zp_len
                    all_death_rate_mean[zp_i][zp_j] = zp_death_rate
                    if log_case == 'death_rate':
                        print('%-6.3f' % zp_death_rate, end=' ')
                    elif log_case == 'cost':
                        print('%-6.3f' % zp_cost, end=' ')
                    else:
                        print('%-6.3f' % zp_len, end=' ')
                if zp_i == param1_len - 1:
                    print('\n', end='')
                else:
                    print('\n', end='')


            print('------------------------------analyze-------------------------------')
            print(
                'death_rate_analyze: \n<=2 -> %-8.4f std-> %-8.4f\n<=10 -> %-8.4f std-> %-8.4f\n<=30 -> %-8.4f std-> %-8.4f\n<=50 ->%-8.4f std-> %-8.4f' % (
                    np.mean(cum_less1_death_rate), np.std(cum_less1_death_rate), np.mean(cum_less2_death_rate),
                    np.std(cum_less2_death_rate),
                    np.mean(cum_less3_death_rate), np.std(cum_less3_death_rate), np.mean(cum_less4_death_rate),
                    np.std(cum_less4_death_rate)))
            print('total<=50 ->%-8.4f std-> %-8.4f' % (
            np.mean(cum_less_total_death_rate), np.std(cum_less_total_death_rate)))

            print(
                'cost_analyze: \n>=0.9 -> %-8.4f std-> %-8.4f\n>=0.8 -> %-8.4f std-> %-8.4f\n>=0.7 -> %-8.4f std-> %-8.4f\n>=0.6 -> %-8.4f std-> %-8.4f' % (
                    np.mean(cum_less1_cost), np.std(cum_less1_cost), np.mean(cum_less2_cost), np.std(cum_less2_cost),
                    np.mean(cum_less3_cost), np.std(cum_less3_cost), np.mean(cum_less4_cost), np.std(cum_less4_cost)))
            print('total>=60 ->%-8.4f std-> %-8.4f' % (np.mean(cum_less_total_cost), np.std(cum_less_total_cost)))

            print(
                'len_analyze: \n>=0.95 -> %-8.4f std-> %-8.4f\n>=0.8 -> %-8.4f std-> %-8.4f\n>=0.7 -> %-8.4f std-> %-8.4f\n>=0.6 -> %-8.4f std-> %-8.4f' % (
                    np.mean(cum_less1_len), np.std(cum_less1_len), np.mean(cum_less2_len), np.std(cum_less2_len),
                    np.mean(cum_less3_len), np.std(cum_less3_len), np.mean(cum_less4_len), np.std(cum_less4_len)))
            print('total>=60 ->%-8.4f std-> %-8.4f' % (np.mean(cum_less_total_len), np.std(cum_less_total_len)))





