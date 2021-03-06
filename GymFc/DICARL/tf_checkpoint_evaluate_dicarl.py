import argparse
from pathlib import Path
import os.path
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import gym
from gymfc_nf.envs import *
from gymfc_nf.utils.monitor import CheckpointMonitor
from gymfc_nf.utils.log import make_header
from DICARL_policy import DICARL_Policy

def generate_inputs(num_trials, max_rate, seed):
    inputs = []
    np.random.seed(seed)
    for i in range(num_trials):
        inputs.append(np.random.normal(0, max_rate, size=3))
    return inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Monitor and evaluate Tensorflow checkpoints.")
    # parser.add_argument('ckpt_dir', help="Directory where checkpoints are saved")
    parser.add_argument('--twin', default="./gymfc_nf/twins/nf1/model.sdf",
                        help="File path of the aircraft digitial twin/model SDF.")
    parser.add_argument('--eval-dir', 
                        help="Directory where evaluation logs are saved, if different than default.")
    parser.add_argument('--gym-id', default="gymfc_nf-step-v1")
    parser.add_argument('--num-trials', type=int, default=3)
    # Provide a seed so the same setpoint will be created. Useful for debugging
    # TODO (wfk) Support different policies from different trainers e.g.,
    # from Stable baselines and Tensorforce

    args = parser.parse_args()

    seed = 66
    # Redefine all args, maybe later convert to a class
    gym_id = args.gym_id

    ckpt_dir = '../log/gymfc_nf-step-v1/DICARL/'

    action_clip = True
    noise_flig = True
    noise_mag = 0.1






    model_dir = Path(ckpt_dir).parent
    if noise_flig:
        eval_dir = args.eval_dir if args.eval_dir else os.path.join(model_dir, "dicarl_evaluations_noise")
    else:
        eval_dir = args.eval_dir if args.eval_dir else os.path.join(model_dir, "dicarl_evaluations")

    num_trials = args.num_trials

    env = gym.make(gym_id)
    env.seed(seed)
    env.set_aircraft_model(args.twin)

    a_upperbound = np.ones(len(env.action_space.high))*1.1
    a_lowerbound = -np.ones(len(env.action_space.high))*0.1


    # For evaluation we compute all of the singlet inputs upfront so we get a
    # more accurate comparison.
    # XXX (wfk) This will not work for other environments with mulitple
    # setpoints!
    inputs = generate_inputs(num_trials, env.max_rate, seed)
    def callback(checkpoint_path):

        checkpoint_filename = os.path.split(checkpoint_path)[-1]
        ckpt_eval_dir = os.path.join(eval_dir, checkpoint_filename)
        Path(ckpt_eval_dir).mkdir(parents=True, exist_ok=True)

        print ("Evaluating checkpoint {}".format(checkpoint_path))
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(checkpoint_path + '.meta',
                                               clear_devices=True)
            saver.restore(sess, checkpoint_path)
            pi = DICARL_Policy(sess)

            es = []
            rs = []
            infos = []
            log_header = ""
            for i in range(num_trials):

                pi.reset()
                ob = env.reset()
                # Override the random generatd input in the environment
                # must do this after the reset becuase this is normally where
                # this gets computed.
                env.generated_input = inputs[i]

                if len(log_header) == 0:
                    log_header = make_header(len(ob))

                log_file = os.path.join(ckpt_eval_dir, "trial-{}.csv".format(i))
                print("\t", log_file)

                sim_time = 0
                actual = np.zeros(3)
                zp_num = 0
                logs = []
                each_es = []
                each_info = []
                each_rs = []
                while True:
                    ac = pi.action(ob, env.sim_time, env.angular_rate_sp,
                                   env.imu_angular_velocity_rpy)
                    zp_num = zp_num + 1
                    if action_clip:
                        ac = a_lowerbound + (np.tanh(ac) + 1.) * (a_upperbound - a_lowerbound) / 2

                    if noise_flig:
                        noise = np.random.uniform(-1,1,4)
                        ac = ac + noise * noise_mag
                    # if zp_num%50==0:
                    #     adv = np.random.rand(4)
                    #     ac = ac + adv
                    ob, reward, done,  info = env.step(ac)

                    # TODO (wfk) Should we standardize this log format? We could
                    # use NASA's SIDPAC channel format.
                    log = ([env.sim_time] +
                            ob.tolist() + # The observations are the NN input
                            ac.tolist() + # The actions are the NN output
                            env.imu_angular_velocity_rpy.tolist() + # Angular velocites
                            env.angular_rate_sp.tolist() + #
                            env.y.tolist() + # Y is the output sent to the ESC
                            env.esc_motor_angular_velocity.tolist() +
                            [reward])# The reward that would have been given for the action, can be helpful for debugging

                    e = env.imu_angular_velocity_rpy - env.angular_rate_sp
                    es.append(e)
                    rs.append(reward)
                    logs.append(log)
                    infos.append(info)
                    each_es.append(e)
                    each_rs.append(reward)
                    each_info.append(info)

                    if done:
                        break
                np.savetxt(log_file, logs, delimiter=",", header=log_header)
                print("\tMAE={:.4f} Ave. Reward={:.4f}".format(np.mean(np.abs(each_es)), np.mean(each_rs)))
                print("Reward Scater", np.mean(each_info, axis=0))

            print("\tMAE={:.4f} Ave. Reward={:.4f}".format(np.mean(np.abs(es)), np.mean(rs)))
            print("\n Reward Scater", np.mean(infos, axis=0))

    monitor = CheckpointMonitor(ckpt_dir, callback)
    monitor.start(5)
    env.close()