## Configuration Environment

Here is how to configure the environment for compilation, code generation, and deployment in Matlab/SimuLink.

1、Introduction

https://rflysim.com/zh/

2、Download the installation package

https://rflysim.com/download.html

3、Install the software package

https://rflysim.com/zh/2_Configuration/SoftwareInstallation.html

4、Hardware configuration

https://rflysim.com/zh/2_Configuration/HardwareConfiguration.html



## File Introduction

Here are the simulink files used in the experiment and some neural network parameters.

*SIL_model* is the model during software-in-the-loop testing

*HIL_model* is the model used in hardware-in-the-loop testing

*actual_flight.slx* is the simulink model used in the actual flight test, which can automatically generate deployable embedded code

*network_parameter_1.mat* stores large-angle parameters

*network_parameter_2.mat* stores small angle parameters

*load_parameter* file can read network parameters to matlab workspace

 *read_data* file can read the data collected by the flight control hardware sensor

*NN.bin* and *PID.bin* are the parameters collected using neural network and PID controller respectively

