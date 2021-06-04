"""
Model(Policy) for both protagonist and adversary
"""

import gym
import tensorflow as tf

from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype


class MlpPolicy(object):

    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, tau, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        print('use zpmpl_Pro')
        self.ac_space = ac_space
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.gaussian_fixed_var = gaussian_fixed_var

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        self.ob_ = U.get_placeholder(name="ob_", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter_pro"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        with tf.variable_scope('vf_pro'):
            self.obz = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = self.obz
            for i in range(self.num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, self.hid_size, name="vffc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name="vffinal", kernel_initializer=U.normc_initializer(1.0))[:,0]

        self.pdparam = self.build_action(self.ob)
        self.pdparam_ = self.build_action(self.ob_, reuse=True)

        self.pd = pdtype.pdfromflat(self.pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = self.pd.sample()
        self.ac_, _ = self.sample_()
        self._act = U.function([stochastic, self.ob], [ac, self.vpred])

    def build_action(self, s, name='actor', reuse=None, custom_getter=None):
        if reuse is None:
            trainable = True
            last_out = self.obz
        else:
            trainable = False
            obz = tf.clip_by_value((s - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)  # 这里是利用均值把状态约束在-5,5之间，
            last_out = obz
        # trainable = True
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):

            for i in range(self.num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, self.hid_size, name="polfc%i" % (i + 1),
                                                      kernel_initializer=U.normc_initializer(1.0),
                                                      trainable=trainable))
            if self.gaussian_fixed_var and isinstance(self.ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, self.pdtype.param_shape()[0] // 2, name="polfinal",
                                       kernel_initializer=U.normc_initializer(0.01),
                                       trainable=trainable)
                logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0] // 2],
                                         initializer=tf.zeros_initializer(),
                                         trainable=trainable)
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, self.pdtype.param_shape()[0], name="polfinal",
                                          kernel_initializer=U.normc_initializer(0.01),
                                          trainable=trainable)
        return pdparam

    def sample_(self):
        mean, logstd = tf.split(axis=len(self.pdparam_.shape)-1, num_or_size_splits=2, value=self.pdparam_)
        std = tf.exp(logstd)
        return mean + std * tf.random_normal(tf.shape(mean)), mean

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    def save(self, path):
        U.save_state(path)

    def load(self, path):
        U.load_state(path)
