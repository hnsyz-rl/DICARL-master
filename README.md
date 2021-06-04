# Robust Adversarial Reinforcement Learning with Dissipation Inequation Constraint

This repository is the official implementation of "Robust Adversarial Reinforcement Learning with Dissipation Inequation Constraint". 


<div align=center><img src = "https://github.com/hnsyz-rl/DICARL-master/blob/master/Figure/figure1.jpg" width=600 alt="figure"></div>
<div align=center>Figure 1. Overview of the proposed dissipation-inequation-constraint-based adversarial reinforcement learning (DICARL) method.</div>


## Conda environment
From the general python package sanity perspective, it is a good idea to use conda environments to make sure packages from different projects do not interfere with each other.

To create a conda env with python3, one runs 
```bash
conda create -n dicarl python=3.6
```
To activate the env: 
```
conda activate dicarl
```

## Requirements

To install requirements:
For the MuJoCo environments (Experiment 4.1 and 4.2):
```setup
pip install -r requirements_mujoco.txt
```
Note that the installation of the adversarial environment gym-adv needs to execute the following command:
```
cd gym-adv
pip install -e .
```

For the GymFc environment (Experiment 4.3):
```setup
pip install -r requirements_gymfc.txt
```
For more information on GymFc environment, refer to the [GymFc webpage](https://github.com/wil3/gymfc).

Since this environment uses the Gazebo 10, you'll need to install Gazebo following [this](http://gazebosim.org/).


## Training

To train the experiment 4.1 and 4.2 in the paper, run this command:

```train
cd MuJoCo/InvertedDoublePendulum/code/
python run_Test.py
```
Where the parameter is_train needs to be set to:

```
is_train = True
```

The hyperparameters and the learning algorithm can be changed via changing the `run_Test.py`, for example:

The algorithm name could be one of ['DICARL', 'Domain_rand', 'NR_MDP', 'PPO', 'RARL'], and you can modify the training_name variable to change the different algorithm.

Other hyperparameters are also adjustable in `run_Test.py`. You can select different tasks by switching different folders. For example, test Hopper can switch to:

```train
cd MuJoCo/Hopper/code/
python run_Test.py
```

To train experiment 4.3 in the paper, run follows command:

```train
cd GymFc/DICARL/
python run_Test.py
```

Where the parameter is_train needs to be set to:

```
is_train = True
```

The hyperparameters can be changed via change the `run_Test.py`. You can select different algorithms by switching different folders. For example, test Vanllia PPO can switch to:

```train
cd GymFc/Vanllia_PPO/
python run_Test.py
```

## Evaluation

To evaluate the model for the experiment 4.1 and 4.2, run:

```eval
python run_Test.py
```

Where the parameter is_train needs to be set to:
```
is_train = False
```
Please load the correct model file name. For example, to run our trained model with HalfCheetah environment, you can set test_name to:

```
test_name = 'DICARL-2e6-baselines-action-clip-0326_Lclip_009_linear-adv-3_3'
```

To evaluate the model for experiment 4.3, run:

```eval
cd GymFc/DICARL/
python tf_checkpoint_evaluate_dicarl.py
```

You can test different algorithms by switching different folders and then look at specific evaluation trials using,

```eval
cd GymFc
python plot_flight.py
```

## Trained policies

You can find trained policies here:

For the experiment 4.1 and 4.2,
- /MuJoCo/HalfCheetah/log/HalfCheetahTorsoAdv-v1/
- /MuJoCo/Hopper/log/HopperHeelAdv-v1/
- /MuJoCo/InvertedDoublePendulum/log/InvertedDoublePendulumAdv-v1/
- /MuJoCo/InvertedPendulum/log/InvertedPendulumAdv-v1/

For experiment 4.3,
- /GymFc/log/gymfc_nf-step-v1/

The hyperparameters of all algorithms can be found in Appendix C.2, D, and E.2.

## Results

Our algorithm achieves the following performance:

### Robustness under the modeling error

Table 1. Success rates and standard deviations of different algorithms with 100 mass combinations are compared. The trained policies are initialized by seven random seeds, and 700 episodes are tested for each mass group. Signiﬁcantly better results from a t-test with p < 1% are highlighted in bold.
  
| Algorithm          | InvertedPendulum  | InvertedDoublePendulum | HalfCheetah    |      Hopper      |
| ------------------ |-------------------| ---------------------- | -------------- | --------------   |
| Vanilla PPO        |    59.2±0.4       |      36.3±1.19         |      66.7±2.24 |      14.6±0.49   |
| RARL               |    72.8±0.75      |      40.6±1.62         |      82.5±1.96 |      20.4±0.66   |
| NR-MDP             |    71.2±0.4       |      26.9±1.37         |      46.8±2.18 |      12.9±1.14   |
| Oracle             |    **78.0±0.2**   |      33.3±1.49         |      84.3±1.27 |      22.0±0.45   |
| DICARL(ours)       |    77.1±0.7       |    **44.1±1.87**       |    **87.3±0.9**|    **38.4±1.28** |

<div align=center><img src = "https://github.com/hnsyz-rl/DICARL-master/blob/master/Figure/figure2.jpg"></div>
<div align=center>Figure 2. Average failure rates across seven seeds on each test set. The x- and y-axes represent the mass changes of different parts of the robot, respectively.</div>

### Inﬂuence of robustness constraint on system stability

<div align=center><img src = "https://github.com/hnsyz-rl/DICARL-master/blob/master/Figure/figure3.jpg"></div>
<div align=center>Figure 3. Success rates and thrice the standard deviations of DICARL and RARL in different adversary magnitudes are compared. The trained policies are initialized by seven random seeds, and 700 episodes are tested for each mass group.</div>

### Sim-to-real task

<div align=center><img src = "https://github.com/hnsyz-rl/DICARL-master/blob/master/Figure/figure4.jpg"></div>
<div align=center>Figure 4. Step responses and control signals of the three algorithms in the GymFc training environment. OS is short for overshoot, whereas blue line represents the actual aircraft angular velocity, and dashed black line represents the desired angular velocity.</div>

<div align=center><img src = "https://github.com/hnsyz-rl/DICARL-master/blob/master/Figure/figure5.jpg"></div>
<div align=center>Figure 5. Trajectories of the quadcopter ﬂying over the racing gate carrying payloads of different masses. Where the blue trajectory does not carry a payload, the yellow trajectory carries a payload of 63.3 g, and the red trajectory carries a payload of 98.68 g.</div>

Table 2. The normalized average error of pitch, roll, and yaw axes of DICARL and PID attitude controllers in the real world. Signiﬁcantly better results from a t-test with p < 1% are highlighted in bold.

| Controller          | Pitch error  | Roll error | Yaw error    |   
| ------------------ |-------------------| ---------------------- | -------------- | 
| PID        |    0.3595       |      0.3168         |      **0.0514** |  
| DICARL               |    **0.2206**      |      **0.1158**         |      0.0721 |  


<div align=center><img src = "https://github.com/hnsyz-rl/DICARL-master/blob/master/Figure/figure6.jpg"></div>
<div align=center>Figure 6. Flight data of DICARL and PID controller. Black curve shows the remote-control input, whereas the red curve shows the angular velocity measured by the IMU. The ﬂight controller sampling frequency is 0.01s, whereas the sampling duration is 50s.</div>


### Video

<div align=center><img src = "https://github.com/hnsyz-rl/DICARL-master/blob/master/Figure/video1.gif" width=600 alt="figure"></div>
<div align=center>Visulization of the quadcopter running the DICARL controller ﬂying over the racing gate and carrying payloads of different masses.</div>



