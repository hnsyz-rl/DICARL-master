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

def traj_segment_generator(pro_pi, adv_pi, env, alpha, horizon, max_steps_episode, stochastic, clip_action=True):
    t = 0
    # ac = env.sample_action()  # not used, just so we have the datatype
    pro_ac = env.action_space.sample()
    adv_ac = env.action_space.sample()

    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    a_upperbound = np.ones(len(env.action_space.high))*1.1
    a_lowerbound = -np.ones(len(env.action_space.high))*0.1

    # d_upperbound = a_upperbound
    # d_lowerbound = a_lowerbound

    d_upperbound = np.ones(len(env.action_space.high))
    d_lowerbound = -d_upperbound

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
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

        # ac.pro = pro_ac
        # ac.adv = adv_ac
        if clip_action:
            pro = a_lowerbound + (np.tanh(pro_ac) + 1.) * (a_upperbound - a_lowerbound) / 2
            adv = d_lowerbound + (np.tanh(adv_ac) + 1.) * (d_upperbound - d_lowerbound) / 2
        else:
            pro = pro_ac
            adv = adv_ac
        ac = (1-alpha) * pro + alpha * adv
        # ac.adv = 0
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value

        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "pro_vpred": pro_vpreds, "adv_vpred": adv_vpreds, "new": news,
                    "pro_ac": pro_acs, "adv_ac": adv_acs, "pro_prevac": pro_prevacs, "adv_prevac": adv_prevacs,
                    "pro_nextvpred": pro_vpred * (1 - new), "adv_nextvpred": adv_vpred * (1 - new),
                    "ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
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

        rew = rew_pre
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1

        # if cur_ep_len > max_steps_episode:
        #     new = True
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1



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
        adv_delta = -rew[t] - gamma * adv_vpred[t+1] * nonterminal + adv_vpred[t]
        pro_gaelam[t] = pro_lastgaelam = pro_delta + gamma * lam * nonterminal * pro_lastgaelam
        adv_gaelam[t] = adv_lastgaelam = adv_delta + gamma * lam * nonterminal * adv_lastgaelam
    seg["pro_tdlamret"] = seg["pro_adv"] + seg["pro_vpred"]
    seg["adv_tdlamret"] = seg["adv_adv"] + seg["adv_vpred"]


def learn(env, policy_func, policy_fn_adv, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize, max_steps_episode,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        clip_action = True, alpha=0.1,
        ckpt_dir=None,
        restore_dir=None,
        save_timestep_period=1e7
        ):
    # Setup losses and stuff
    # ----------------------------------------
    rew_mean = []

    ob_space = env.observation_space
    pro_ac_space = env.action_space
    adv_ac_space = env.adv_action_space

    pro_pi = policy_func("pro_pi", ob_space, pro_ac_space) # Construct network for new policy
    pro_oldpi = policy_func("pro_oldpi", ob_space, pro_ac_space) # Network for old policy
    adv_pi = policy_fn_adv("adv_pi", ob_space, adv_ac_space) # Construct network for new adv policy
    adv_oldpi = policy_fn_adv("adv_oldpi", ob_space, adv_ac_space) # Network for old adv policy

    pro_atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    adv_atarg = tf.placeholder(dtype=tf.float32, shape=[None])

    ret_pro = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    ret_adv = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon #?????????????????????????????????????????????

    ob = U.get_placeholder_cached(name="ob")
    ob_adv = U.get_placeholder_cached(name="ob_adv")

    pro_ac = pro_pi.pdtype.sample_placeholder([None])
    adv_ac = adv_pi.pdtype.sample_placeholder([None])

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

    pro_ratio = tf.exp(pro_pi.pd.logp(pro_ac) - pro_oldpi.pd.logp(pro_ac)) # pnew / pold
    adv_ratio = tf.exp(adv_pi.pd.logp(adv_ac) - adv_oldpi.pd.logp(adv_ac))
    pro_surr1 = pro_ratio * pro_atarg # surrogate from conservative policy iteration
    adv_surr1 = adv_ratio * adv_atarg
    pro_surr2 = tf.clip_by_value(pro_ratio, 1.0 - clip_param, 1.0 + clip_param) * pro_atarg #
    adv_surr2 = tf.clip_by_value(adv_ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_atarg
    pro_pol_surr = - tf.reduce_mean(tf.minimum(pro_surr1, pro_surr2)) # PPO's pessimistic surrogate (L^CLIP)
    adv_pol_surr = - tf.reduce_mean(tf.minimum(adv_surr1, adv_surr2))
    pro_vf_loss = tf.reduce_mean(tf.square(pro_pi.vpred - ret_pro))
    adv_vf_loss = tf.reduce_mean(tf.square(adv_pi.vpred - ret_adv))
    pro_total_loss = pro_pol_surr + pro_pol_entpen + pro_vf_loss
    adv_total_loss = adv_pol_surr + adv_pol_entpen + adv_vf_loss
    pro_losses = [pro_pol_surr, pro_pol_entpen, pro_vf_loss, pro_meankl, pro_meanent]
    pro_loss_names = ["pro_pol_surr", "pro_pol_entpen", "pro_vf_loss", "pro_kl", "pro_ent"]
    adv_losses = [adv_pol_surr, adv_pol_entpen, adv_vf_loss, adv_meankl, adv_meanent]
    adv_loss_names = ["adv_pol_surr", "adv_pol_entpen", "adv_vf_loss", "adv_kl", "adv_ent"]

    pro_var_list = pro_pi.get_trainable_variables()
    adv_var_list = adv_pi.get_trainable_variables()
    pro_lossandgrad = U.function([ob, pro_ac, pro_atarg, ret_pro, lrmult], pro_losses + [U.flatgrad(pro_total_loss, pro_var_list)])
    adv_lossandgrad = U.function([ob_adv, adv_ac, adv_atarg, ret_adv, lrmult], adv_losses + [U.flatgrad(adv_total_loss, adv_var_list)])
    pro_adam = MpiAdam(pro_var_list, epsilon=adam_epsilon)
    adv_adam = MpiAdam(adv_var_list, epsilon=adam_epsilon)

    pro_assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(pro_oldpi.get_variables(), pro_pi.get_variables())])
    adv_assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(adv_oldpi.get_variables(), adv_pi.get_variables())])
    pro_compute_losses = U.function([ob, pro_ac, pro_atarg, ret_pro, lrmult], pro_losses)
    adv_compute_losses = U.function([ob_adv, adv_ac, adv_atarg, ret_adv, lrmult], adv_losses)


    saver = None
    if ckpt_dir:
        # Store for each one
        keep = int(max_timesteps/float(save_timestep_period))
        print ("[INFO] Keeping ", keep, " checkpoints")
        saver = tf.train.Saver(save_relative_paths=True, max_to_keep=keep)


    U.initialize()
    pro_adam.sync()
    adv_adam.sync()

    if restore_dir:
        ckpt = tf.train.get_checkpoint_state(restore_dir)
        if ckpt:
            # If there is one that already exists then restore it
            print("Restoring model from ", ckpt.model_checkpoint_path)
            saver.restore(tf.get_default_session(), ckpt.model_checkpoint_path)
        else:
            print("Trying to restore model from ", restore_dir, " but doesn't exist")


    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pro_pi, adv_pi, env, alpha, timesteps_per_batch, max_steps_episode, stochastic=True, clip_action=clip_action)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    print('Version-v7-NR_MDP-PPO')

    # with tf.Session() as sess:  # ????????????????????????
    #     # ???????????????????????????????????????????????????????????????????????????????????????????????????
    #     writer = tf.summary.FileWriter('/home/jianghan/My_Project/DISCARL-PPO/Hopper/log/Graph/RARL3/', sess.graph)
    #     sess.run(tf.global_variables_initializer())
    #     # f = sess.run(output)
    #     # print(f)
    #     writer.close()
    #     pass

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
            task_name = "ppo1-{}-{}.ckpt".format(env.spec.id, timesteps_so_far)
            fname = os.path.join(ckpt_dir, task_name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver.save(tf.get_default_session(), fname)
            next_ckpt_timestep += save_timestep_period

        if end:
            break


        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        Percentage = min(timesteps_so_far/max_timesteps, 1) * 100
        # logger.log("**********Iteration %i **Percentage %.2f **********" % (iters_so_far, Percentage))

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, pro_ac, adv_ac, pro_atarg, adv_atarg, pro_tdlamret, adv_tdlamret = seg["ob"], seg["pro_ac"], seg["adv_ac"], seg["pro_adv"], seg["adv_adv"], seg["pro_tdlamret"], seg["adv_tdlamret"]
        pro_vpredbefore = seg["pro_vpred"] # predicted value function before udpate
        adv_vpredbefore = seg["adv_vpred"]
        pro_atarg = (pro_atarg - pro_atarg.mean()) / pro_atarg.std() # standardized advantage function estimate
        adv_atarg = (adv_atarg - adv_atarg.mean()) / adv_atarg.std()

        # TODO
        d = Dataset(dict(ob=ob, ac=pro_ac, atarg=pro_atarg, vtarg=pro_tdlamret), shuffle=not pro_pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pro_pi, "ob_rms"):
            pro_pi.ob_rms.update(ob) # update running mean/std for policy

        pro_assign_old_eq_new() # set old parameter values to new parameter values
        # logger.log("Pro Optimizing...")
        # logger.log(fmt_row(13, pro_loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            pro_losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = pro_lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                pro_adam.update(g, optim_stepsize * cur_lrmult)
                pro_losses.append(newlosses)

        # pro_losses = []
        # for batch in d.iterate_once(optim_batchsize):
        #     newlosses = pro_compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
        #     pro_losses.append(newlosses)
        # pro_meanlosses,_,_ = mpi_moments(pro_losses, axis=0)
        # logger.log(fmt_row(13, pro_meanlosses))

        d = Dataset(dict(ob=ob, ac=adv_ac, atarg=adv_atarg, vtarg=adv_tdlamret), shuffle=not adv_pi.recurrent)
        if hasattr(adv_pi, "ob_rms"):
            adv_pi.ob_rms.update(ob)

        adv_assign_old_eq_new()

        # ???????????????????????????
        # logger.log("Adv Optimizing...")
        # logger.log(fmt_row(13, adv_loss_names))
        # logger.log(fmt_row(13, adv_loss_names))
        for _ in range(optim_epochs):
            adv_losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = adv_lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adv_adam.update(g, optim_stepsize * cur_lrmult)
                adv_losses.append(newlosses)

        adv_losses = []
        # for batch in d.iterate_once(optim_batchsize):
        #     newlosses = adv_compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
        #     adv_losses.append(newlosses)
        # adv_meanlosses,_,_ = mpi_moments(adv_losses, axis=0)
        # logger.log(fmt_row(13, adv_meanlosses))

        # curr_rew = evaluate(pro_pi, test_env)
        # rew_mean.append(curr_rew)
        # print(curr_rew)

        # logger.record_tabular("ev_tdlam_before", explained_variance(pro_vpredbefore, pro_tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)

        # logger.dump_tabular()


    return pro_pi, np.array(rew_mean), np.mean(lenbuffer), timesteps_so_far


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
