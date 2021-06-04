from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
import scipy.io as sio

model_dir1 = '/home/len/Data/Project/Robust_Gymfc/GYmfc-DISCARL/examples/log/To_real/ppo-gymfc_nf-step-v1-2101248.ckpt'
reader = pywrap_tensorflow.NewCheckpointReader(model_dir1)
var = reader.get_variable_to_shape_map()

for key in var:
    if(key.startswith("pro_pi/")):
        print("tensor_name1: ", key)
       # print(reader.get_tensor(key))


params = {}
name = ['sum','sumsq','count']

for t, key in enumerate(var):

    if key.startswith("pro_pi/obfilter_pro/runningsum"):
        params['sum'] = reader.get_tensor(key)


    if key.startswith("pro_pi/obfilter_pro/runningsumsq"):
        params['sumsq'] = reader.get_tensor(key)

    if key.startswith("pro_pi/obfilter_pro/count"):
        params['count'] = reader.get_tensor(key)

    if key.startswith("pro_pi/actor/polfc1/kernel"):
        params['layer9'] = reader.get_tensor(key)

    if key.startswith("pro_pi/actor/polfc1/bias"):
        params['layer10'] = reader.get_tensor(key)

    if key.startswith("pro_pi/actor/polfc2/kernel"):
        params['layer11'] = reader.get_tensor(key)

    if key.startswith("pro_pi/actor/polfc2/bias"):
        params['layer12'] = reader.get_tensor(key)

    if key.startswith("pro_pi/actor/polfinal/kernel"):
        params['layer13'] = reader.get_tensor(key)

    if key.startswith("pro_pi/actor/polfinal/bias"):
        params['layer14'] = reader.get_tensor(key)



obj_arr = np.zeros((1,), dtype=np.object)
obj_arr[0] = params
sio.savemat('checkpoint.mat', {'results': obj_arr})
print("--------------------------------------------------")
print(params)
