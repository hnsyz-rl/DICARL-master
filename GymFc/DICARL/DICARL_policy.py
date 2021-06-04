import numpy as np
import tensorflow as tf
from gymfc_nf.policies.policy import Policy
class DICARL_Policy(Policy):
    def __init__(self, sess):
        print('\nAlgorithm name: DICARL\n')
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name('pro_pi/ob:0') 
        self.y = graph.get_tensor_by_name('pro_pi/actor/polfinal/BiasAdd:0')
        self.sess = sess

    def action(self, state, sim_time=0, desired=np.zeros(3), actual=np.zeros(3) ):

        y_out = self.sess.run(self.y, feed_dict={self.x:[state] })
        return y_out[0]
