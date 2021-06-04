"""
RARL Framework
"""

import time
from collections import deque
import tensorflow as tf, numpy as np
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
import os.path
from copy import deepcopy


def traj_segment_generator(pro_pi, adv_pi, env, horizon, max_steps_episode, stochastic, clip_action=False,):
    t = 0
    ac = env.sample_action()  # not used, just so we have the datatype
    pro_ac = ac.pro
    adv_ac = ac.adv

    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low

    d_upperbound = env.adv_action_space.high
    d_lowerbound = env.adv_action_space.low

    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_cost = 0
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    ep_cost = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    obs_ = np.array([ob for _ in range(horizon)])

    rews = np.zeros(horizon, 'float32')
    # print(rews)
    rews_costs = np.zeros(horizon, 'float32')

    pro_vpreds = np.zeros(horizon, 'float32')
    adv_vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    pro_acs = np.array([pro_ac for _ in range(horizon)])
    adv_acs = np.array([adv_ac for _ in range(horizon)])
    pro_prevacs = pro_acs.copy()
    adv_prevacs = adv_acs.copy()

    while True:
        pro_prevac = pro_ac
        adv_prevac = adv_ac
        pro_ac, pro_vpred = pro_pi.act(stochastic, ob)
        adv_ac, adv_vpred = adv_pi.act(stochastic, ob)
        # ac = pro_ac + adv_ac * 0.1
        # action = (np.tanh(pro_ac) * 0.5 + 0.5) + (np.tanh(adv_ac) * 0.5 + 0.5) * 0.05
        if clip_action:
            ac.pro = a_lowerbound + (np.tanh(pro_ac) + 1.) * (a_upperbound - a_lowerbound) / 2
            ac.adv = d_lowerbound + (np.tanh(adv_ac) + 1.) * (d_upperbound - d_lowerbound) / 2
        else:
            ac.pro = pro_ac
            ac.adv = adv_ac
        # print(ac.adv)
        # ac.adv = ac.adv * 0
        # ac.pro = pro_ac
        # ac.adv = adv_ac


        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value

        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "ob_": obs_, "rew": rews, "pro_vpred": pro_vpreds, "adv_vpred": adv_vpreds, "new": news,
                    "pro_ac": pro_acs, "adv_ac": adv_acs, "pro_prevac": pro_prevacs, "adv_prevac": adv_prevacs,
                    "pro_nextvpred": pro_vpred * (1 - new), "adv_nextvpred": adv_vpred * (1 - new),
                    "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_cost": ep_cost}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_cost = []

        i = t % horizon
        obs[i] = ob
        pro_vpreds[i] = pro_vpred
        adv_vpreds[i] = adv_vpred
        news[i] = new
        pro_acs[i] = pro_ac
        adv_acs[i] = adv_ac
        pro_prevacs[i] = pro_prevac
        adv_prevacs[i] = adv_prevac

        ob, rew_pre, new, _ = env.step(ac)

        obs_[i] = ob
        rew = -rew_pre
        # rews_cost = rew_pre[1]
        rews[i] = rew
        # print(rews)

        cur_ep_ret += rew
        cur_ep_len += 1
        # cur_ep_cost += rew_pre[0]

        if cur_ep_len > max_steps_episode:
            new = True

        if new:
            # print('ob', ob, 'rew', rew_pre, 'done', new)
            # print('action', ac.pro, ac.adv)
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            ep_cost.append(cur_ep_cost)

            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_cost = 0
            ob = env.reset()
        t += 1


def evaluate(pi, env, rollouts=50):
    retlst = []
    for _ in range(rollouts):
        ret = 0
        # t = 1e5
        ob = env.reset()
        done = False
        # while not done and t > 0:
        while not done:
            ac = pi.act(False, ob)[0]
            ob, rew, done, _ = env.step(ac)
            ret += rew
            # t -= 1
        retlst.append(ret)
    return np.mean(retlst)


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    pro_vpred = np.append(seg["pro_vpred"], seg["pro_nextvpred"])
    adv_vpred = np.append(seg["adv_vpred"], seg["adv_nextvpred"])
    T = len(seg["rew"])
    seg["pro_adv"] = pro_gaelam = np.empty(T, 'float32')
    seg["adv_adv"] = adv_gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    pro_lastgaelam = 0
    adv_lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        pro_delta = rew[t] + gamma * pro_vpred[t+1] * nonterminal - pro_vpred[t]
        # adv_delta = -rew[t] - gamma * pro_vpred[t+1] * nonterminal + pro_vpred[t]
        adv_delta = -rew[t] - gamma * adv_vpred[t+1] * nonterminal + adv_vpred[t]

        pro_gaelam[t] = pro_lastgaelam = pro_delta + gamma * lam * nonterminal * pro_lastgaelam
        adv_gaelam[t] = adv_lastgaelam = adv_delta + gamma * lam * nonterminal * adv_lastgaelam
    seg["pro_tdlamret"] = seg["pro_adv"] + seg["pro_vpred"]
    seg["adv_tdlamret"] = seg["adv_adv"] + seg["adv_vpred"]



def sample_next_act(ob, a_dim, policy, stochastic):
    T = ob.shape[0]
    next_act = np.zeros([T, a_dim])
    for i in range(T):
        next_act[i][:], _ = policy.act(stochastic, ob[i][:])
    return next_act

def build_l(s, a, s_dim, a_dim,reuse=None, custom_getter=None):
    trainable = True if reuse is None else False
    with tf.variable_scope('Lyapunov', reuse=reuse, custom_getter=custom_getter):
        n_l1 = 64  # 30 # 正儿八经用的时候L网络又和Q网络的参数不一样了，bitch
        w1_s = tf.get_variable('w1_s', [s_dim, n_l1], trainable=trainable)
        w1_a = tf.get_variable('w1_a', [a_dim, n_l1], trainable=trainable)
        b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
        net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
        net_1 = tf.layers.dense(net_0, 64, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
        # net_2 = tf.layers.dense(net_1, 32, activation=tf.nn.relu, name='l3', trainable=trainable)  # 原始是30
        return tf.layers.dense(net_1, 1, trainable=trainable)  # L(s,a)

def learn(env, policy_func_pro, policy_func_adv, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, lr_l, lr_a, max_steps_episode,# advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        clip_action = False,
        restore_dir = None,
        ckpt_dir = None,
        save_timestep_period = 1000
        ):
    # Setup losses and stuff
    # ----------------------------------------
    rew_mean = []

    ob_space = env.observation_space
    pro_ac_space = env.action_space
    adv_ac_space = env.adv_action_space

    # env.render()
    pro_pi = policy_func_pro("pro_pi", ob_space, pro_ac_space) # Construct network for new policy
    pro_oldpi = policy_func_pro("pro_oldpi", ob_space, pro_ac_space) # Network for old policy

    adv_pi = policy_func_adv("adv_pi", ob_space, adv_ac_space) # Construct network for new adv policy
    adv_oldpi = policy_func_adv("adv_oldpi", ob_space, adv_ac_space) # Network for old adv policy

    pro_atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    adv_atarg = tf.placeholder(dtype=tf.float32, shape=[None])

    ret_pro = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    ret_adv = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ob_ = U.get_placeholder_cached(name="ob_")

    adv_ob = U.get_placeholder_cached(name="ob_adv")
    adv_ob_ = U.get_placeholder_cached(name="adv_ob_")


    pro_ac = pro_pi.pdtype.sample_placeholder([None])
    adv_ac = adv_pi.pdtype.sample_placeholder([None])


    # define Lyapunov net
    s_dim = ob_space.shape[0]
    a_dim = pro_ac_space.shape[0]
    d_dim = adv_ac_space.shape[0]

    use_lyapunov = True
    approx_value = True
    finite_horizon = True

    tau = 5e-3
    lr_lambda = 1.0
    lr_alpha = 0.03

    LN_R = tf.placeholder(tf.float32, [None, 1], 'r')  # 回报
    LN_V = tf.placeholder(tf.float32, [None, 1], 'v')  # 回报
    LR_L = tf.placeholder(tf.float32, None, 'LR_L')  # Lyapunov网络学习率
    LR_A = tf.placeholder(tf.float32, None, 'LR_A')  # Actor网络学习率
    L_terminal = tf.placeholder(tf.float32, [None, 1], 'terminal')
    # LN_S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 状态
    # LN_a_input = tf.placeholder(tf.float32, [None, a_dim], 'a_input')  # batch中输入的动作
    # ob_ = tf.placeholder(tf.float32, [None, s_dim], 's_')  # 后继状态
    LN_a_input_ = tf.placeholder(tf.float32, [None, a_dim], 'a_')  # 后继状态
    LN_d_input = tf.placeholder(tf.float32, [None, d_dim], 'd_input')  # batch中输入的干扰
    labda = tf.placeholder(tf.float32, None, 'LR_lambda')

    # log_labda = tf.get_variable('lambda', None, tf.float32, initializer=tf.log(labda_init))  # log(λ)，用于自适应系数
    # labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)

    l = build_l(ob, pro_ac, s_dim, a_dim)  # lyapunov 网络
    l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Lyapunov')

    ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement  网络软更新
    def ema_getter(getter, name, *args, **kwargs):
        return ema.average(getter(name, *args, **kwargs))
    target_update = [ema.apply(l_params)]

    a_ = pro_pi.ac_
    a_old_ = pro_oldpi.ac_

    # 这里的下一个动作的采样方式有好几种
    l_ = build_l(ob_, a_, s_dim, a_dim, reuse=True)

    l_d = build_l(adv_ob, pro_ac, s_dim, a_dim, reuse=True)
    l_d_ = build_l(adv_ob_, LN_a_input_, s_dim, a_dim, reuse=True)

    l_old_ = build_l(ob_, a_old_, s_dim, a_dim, reuse=True, custom_getter=ema_getter)  # 这里是否可以用a_代替a_old呢
    l_derta = tf.reduce_mean(l_ - l + 0.5 * LN_R - lr_lambda * tf.expand_dims(tf.norm(LN_d_input, axis=1), axis=1))
    # d 里的l这一项可能也需要改，因为这里主策略已经优化过了，所以需要重新采样，但是重新采样好像也不太适合，因为Ｒ变了，要不还有一个方式，就是轮流更新
    # 还有一种方式就是这里的ac_也采用样本里的，来保证一致性，但是为啥之前SAC那个就没问题呢
    l_d_derta = tf.reduce_mean(l_d_ - l_d + 0.5 * LN_R - lr_lambda * tf.expand_dims(tf.norm(adv_pi.ac, axis=1), axis=1))  # 可能是这里震荡了, adv_pi.ac  或许这里我真的该同步两个lyapunov
    # labda_loss = -tf.reduce_mean(log_labda * l_derta)  # lambda的更新loss
    # lambda_train = tf.train.AdamOptimizer(LR_A).minimize(labda_loss, var_list=log_labda)  # alpha的优化器

    with tf.control_dependencies(target_update):
        if approx_value:
            if finite_horizon:
                l_target = LN_V  # 这里自己近似会不会好一点
            else:
                l_target = LN_R + gamma * (1 - L_terminal) * tf.stop_gradient(l_old_)  # Lyapunov critic - self.alpha * next_log_pis
        else:
            l_target = LN_R
        l_error = tf.losses.mean_squared_error(labels=l_target, predictions=l)
        ltrain = tf.train.AdamOptimizer(LR_L).minimize(l_error, var_list=l_params)
        Lyapunov_train = [ltrain]
        Lyapunov_opt_input = [LN_R, LN_V, L_terminal, ob, ob_, pro_ac, LN_d_input, LR_A, LR_L]

        Lyapunov_opt = U.function(Lyapunov_opt_input, Lyapunov_train)
        Lyapunov_opt_loss = U.function(Lyapunov_opt_input, [l_error])
    # Lyapunov函数


    pro_kloldnew = pro_oldpi.pd.kl(pro_pi.pd) # compute kl difference
    adv_kloldnew = adv_oldpi.pd.kl(adv_pi.pd)
    pro_ent = pro_pi.pd.entropy()
    adv_ent = adv_pi.pd.entropy()
    pro_meankl = tf.reduce_mean(pro_kloldnew)
    adv_meankl = tf.reduce_mean(adv_kloldnew)
    pro_meanent = tf.reduce_mean(pro_ent)
    adv_meanent = tf.reduce_mean(adv_ent)
    pro_pol_entpen = (-entcoeff) * pro_meanent
    adv_pol_entpen = (-entcoeff) * adv_meanent

    pro_ratio = tf.exp(pro_pi.pd.logp(pro_ac) - pro_oldpi.pd.logp(pro_ac))  # pnew / pold
    adv_ratio = tf.exp(adv_pi.pd.logp(adv_ac) - adv_oldpi.pd.logp(adv_ac))
    pro_surr1 = pro_ratio * pro_atarg  # surrogate from conservative policy iteration
    adv_surr1 = adv_ratio * adv_atarg
    pro_surr2 = tf.clip_by_value(pro_ratio, 1.0 - clip_param, 1.0 + clip_param) * pro_atarg  #
    adv_surr2 = tf.clip_by_value(adv_ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_atarg
    pro_pol_surr = - tf.reduce_mean(tf.minimum(pro_surr1, pro_surr2)) # PPO's pessimistic surrogate (L^CLIP)
    adv_pol_surr = - tf.reduce_mean(tf.minimum(adv_surr1, adv_surr2))
    pro_vf_loss = tf.reduce_mean(tf.square(pro_pi.vpred - ret_pro))
    adv_vf_loss = tf.reduce_mean(tf.square(adv_pi.vpred - ret_adv))
    pro_lyapunov_loss = tf.reduce_mean(-l_derta * labda)
    pro_total_loss = pro_pol_surr + pro_pol_entpen + pro_vf_loss + pro_lyapunov_loss

    adv_lyapunov_loss = tf.reduce_mean(-l_d_derta * labda)
    adv_total_loss = adv_pol_surr + adv_pol_entpen + adv_vf_loss + adv_lyapunov_loss

    pro_losses = [pro_pol_surr, pro_pol_entpen, pro_vf_loss, pro_meankl, pro_meanent, pro_lyapunov_loss]
    pro_loss_names = ["pro_pol_surr", "pro_pol_entpen", "pro_vf_loss", "pro_kl", "pro_ent", "pro_lyapunov_loss"]
    adv_losses = [adv_pol_surr, adv_pol_entpen, adv_vf_loss, adv_meankl, adv_meanent, adv_lyapunov_loss]
    adv_loss_names = ["adv_pol_surr", "adv_pol_entpen", "adv_vf_loss", "adv_kl", "adv_ent", "adv_lyapunov_loss"]

    pro_var_list = pro_pi.get_trainable_variables()
    adv_var_list = adv_pi.get_trainable_variables()
    pro_opt_input = [ob, pro_ac, pro_atarg, ret_pro, lrmult, LN_R, LN_V, ob_, LN_d_input, LR_A, LR_L, labda]
    # pro_lossandgrad = U.function([ob, pro_ac, pro_atarg, ret, lrmult], pro_losses + [U.flatgrad(pro_total_loss, pro_var_list)])
    pro_lossandgrad = U.function(pro_opt_input, pro_losses + [U.flatgrad(pro_total_loss, pro_var_list)])

    # Lyapunov_grad = U.function(Lyapunov_opt_input, U.flatgrad(l_derta * labda, pro_var_list))

    adv_opt_input = [adv_ob, adv_ac, adv_atarg, ret_adv, lrmult, LN_R, adv_ob_, pro_ac, LN_a_input_, LR_A, LR_L, labda]
    adv_lossandgrad = U.function(adv_opt_input, adv_losses + [U.flatgrad(adv_total_loss, adv_var_list)])
    pro_adam = MpiAdam(pro_var_list, epsilon=adam_epsilon)
    adv_adam = MpiAdam(adv_var_list, epsilon=adam_epsilon)

    pro_assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(pro_oldpi.get_variables(), pro_pi.get_variables())])
    adv_assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(adv_oldpi.get_variables(), adv_pi.get_variables())])
    pro_compute_losses = U.function(pro_opt_input, pro_losses)
    adv_compute_losses = U.function(adv_opt_input, adv_losses)

    # zp gymfc new
    saver = None
    if ckpt_dir:  # save model
        # Store for each one
        keep = int(max_timesteps/float(save_timestep_period))  # number of model want to save
        print ("[INFO] Keeping ", keep, " checkpoints")
        saver = tf.train.Saver(save_relative_paths=True, max_to_keep=keep)

    print('version: use dicarl v2-v5')
    print('info:', 'lambda', lr_lambda, 'alpha', lr_alpha,  'Finite_horizon',finite_horizon, 'adv_mag', env.adv_action_space.high, 'timesteps', max_timesteps)

    U.initialize()
    pro_adam.sync()
    adv_adam.sync()

    if restore_dir:  # restore model
        ckpt = tf.train.get_checkpoint_state(restore_dir)
        if ckpt:
            # If there is one that already exists then restore it
            print("Restoring model from ", ckpt.model_checkpoint_path)
            saver.restore(tf.get_default_session(), ckpt.model_checkpoint_path)
        else:
            print("Trying to restore model from ", restore_dir, " but doesn't exist")


    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pro_pi, adv_pi, env, timesteps_per_batch, max_steps_episode, stochastic=True, clip_action=clip_action)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    stop_buffer = deque(maxlen=30)
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    costbuffer = deque(maxlen=100)
    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"



    next_ckpt_timestep = save_timestep_period




    while True:
        if callback: callback(locals(), globals())

        end = False
        if max_timesteps and timesteps_so_far >= max_timesteps:
            end = True
        elif max_episodes and episodes_so_far >= max_episodes:
            end = True
        elif max_iters and iters_so_far >= max_iters:
            end = True
        elif max_seconds and time.time() - tstart >= max_seconds:
            end = True

        if saver and ((timesteps_so_far >= next_ckpt_timestep) or end):
            task_name = "ppo-{}-{}.ckpt".format(env.spec.id, timesteps_so_far)
            fname = os.path.join(ckpt_dir, task_name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver.save(tf.get_default_session(), fname)
            next_ckpt_timestep += save_timestep_period

        if end: #and np.mean(stop_buffer) > zp_max_step:
            break

        if end and max_timesteps < 100:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        # 这里是对应的Lyapunov函数的学习率
        Lya_epsilon = 1.0 - (timesteps_so_far - 1.0) / max_timesteps
        if end:
            Lya_epsilon = 0.0000001
        lr_a_this = lr_a * Lya_epsilon
        lr_l_this = lr_l * Lya_epsilon
        lr_alpha_this = lr_alpha * Lya_epsilon

        Percentage = min(timesteps_so_far/max_timesteps, 1) * 100
        logger.log("**********Iteration %i **Percentage %.2f **********" % (iters_so_far, Percentage))



        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, pro_ac, adv_ac, pro_atarg, adv_atarg, pro_tdlamret, adv_tdlamret = seg["ob"], seg["pro_ac"], seg["adv_ac"], seg["pro_adv"], seg["adv_adv"], seg["pro_tdlamret"], seg["adv_tdlamret"]
        rew = seg["rew"]
        ob_ = seg["ob_"]
        new = seg["new"]

        pro_vpredbefore = seg["pro_vpred"] # predicted value function before udpate
        adv_vpredbefore = seg["adv_vpred"]
        pro_atarg = (pro_atarg - pro_atarg.mean()) / pro_atarg.std() # standardized advantage function estimate
        adv_atarg = (adv_atarg - adv_atarg.mean()) / adv_atarg.std()

        # TODO
        # d = Dataset(dict(ob=ob, ac=pro_ac, atarg=pro_atarg, vtarg=pro_tdlamret), shuffle=not pro_pi.recurrent)
        d = Dataset(dict(ob=ob, ob_=ob_, rew=rew, new=new, ac=pro_ac, adv=adv_ac, atarg=pro_atarg, vtarg=pro_tdlamret), shuffle=not pro_pi.recurrent)  # 放入经验回放寄存器
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pro_pi, "ob_rms"): pro_pi.ob_rms.update(ob) # update running mean/std for policy

        pro_assign_old_eq_new() # set old parameter values to new parameter values

        logger.log("Pro Optimizing...")
        logger.log(fmt_row(13, pro_loss_names))

        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            pro_losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                l_value = deepcopy(batch["atarg"])
                zp_fuk = l_value.reshape(-1,1)

                # [LN_R, LN_V, ob, ob_, pro_ac, LN_d_input, LR_A, LR_L]
                Lyapunov_opt(batch["rew"].reshape(-1,1), l_value.reshape(-1,1), batch["new"], batch["ob"], batch["ob_"], batch["ac"],
                             batch["adv"], lr_a_this, lr_l_this)
                *newlosses, g = pro_lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult,
                                                batch["rew"].reshape(-1,1), l_value.reshape(-1,1), batch["ob_"],
                                                batch["adv"], lr_a_this, lr_l_this, lr_alpha_this)

                pro_adam.update(g, optim_stepsize * cur_lrmult)
                pro_losses.append(newlosses)
            # logger.log(fmt_row(13, np.mean(pro_losses, axis=0)))

        # logger.log("Pro Evaluating losses...")
        pro_losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = pro_compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult,
                                                batch["rew"].reshape(-1,1), l_value.reshape(-1,1), batch["ob_"],
                                                batch["adv"], lr_a_this, lr_l_this, lr_alpha_this)
            pro_losses.append(newlosses)
        pro_meanlosses,_,_ = mpi_moments(pro_losses, axis=0)

        logger.log(fmt_row(13, pro_meanlosses))

        ac_ = sample_next_act(ob_, a_dim, pro_pi, stochastic=True)  # ob, a_dim, policy, stochastic

        d = Dataset(dict(ob=ob, adv_ac=adv_ac, atarg=adv_atarg, vtarg=adv_tdlamret, ob_=ob_, rew=rew, ac_=ac_, new=new, pro_ac=pro_ac), shuffle=not adv_pi.recurrent)

        if hasattr(adv_pi, "ob_rms"): adv_pi.ob_rms.update(ob)
        adv_assign_old_eq_new()

        # logger.log(fmt_row(13, adv_loss_names))
        logger.log("Adv Optimizing...")
        logger.log(fmt_row(13, adv_loss_names))
        for _ in range(optim_epochs):
            adv_losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                # adv_opt_input = [ob, adv_ac, adv_atarg, ret, lrmult, LN_R, ob_, LN_a_input_, LR_A, LR_L]
                # [ob, adv_ac, adv_atarg, ret, lrmult, LN_R, ob_, pro_ac, LN_a_input_, LR_A, LR_L]
                # ac_ = sample_next_act(batch["ob_"], a_dim, pro_pi, stochastic=True) # ob, a_dim, policy, stochastic
                *newlosses, g = adv_lossandgrad(batch["ob"], batch["adv_ac"], batch["atarg"], batch["vtarg"], cur_lrmult,
                                                batch["rew"].reshape(-1,1), batch["ob_"],
                                                batch["pro_ac"], batch["ac_"], lr_a_this, lr_l_this, lr_alpha_this)
                # *newlosses, g = adv_lossandgrad(batch["ob"], batch["adv_ac"], batch["atarg"], batch["vtarg"], cur_lrmult,
                #                                 batch["rew"].reshape(-1,1), batch["ob_"],
                #                                 batch["pro_ac"], batch["pro_ac"], lr_a_this, lr_l_this)
                adv_adam.update(g, optim_stepsize * cur_lrmult)
                adv_losses.append(newlosses)
            # logger.log(fmt_row(13, np.mean(adv_losses, axis=0)))
        # logger.log("Adv Evaluating losses...")
        adv_losses = []

        for batch in d.iterate_once(optim_batchsize):
            newlosses = adv_compute_losses(batch["ob"], batch["adv_ac"], batch["atarg"], batch["vtarg"], cur_lrmult,
                                                batch["rew"].reshape(-1,1), batch["ob_"],
                                                batch["pro_ac"], batch["ac_"], lr_a_this, lr_l_this, lr_alpha_this)
            adv_losses.append(newlosses)
        adv_meanlosses,_,_ = mpi_moments(adv_losses, axis=0)
        logger.log(fmt_row(13, adv_meanlosses))

        # curr_rew = evaluate(pro_pi, test_env)
        # rew_mean.append(curr_rew)
        # print(curr_rew)

        # logger.record_tabular("ev_tdlam_before", explained_variance(pro_vpredbefore, pro_tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))

        # print(rews)

        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        cost = seg["ep_cost"]
        costbuffer.extend(cost)

        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpCostMean", np.mean(costbuffer))
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)

        stop_buffer.extend(lens)

        logger.record_tabular("stop_flag", np.mean(stop_buffer))
        logger.dump_tabular()

        # print(stop_buffer)
        print(lr_alpha_this)


    print('version: use dicarl v2-v5')
    print('info:', 'lambda', lr_lambda, 'alpha', lr_alpha,  'Finite_horizon',finite_horizon, 'adv_mag', env.adv_action_space.high, 'timesteps', max_timesteps)

    return pro_pi, np.mean(rewbuffer), timesteps_so_far, np.mean(lenbuffer)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
