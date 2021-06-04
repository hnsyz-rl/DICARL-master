load network_parameter_1.mat;
layer1 = double(results{1,1}.layer9);
layer2 = double(results{1,1}.layer10);
layer3 = double(results{1,1}.layer11);
layer4 = double(results{1,1}.layer12);
layer5 = double(results{1,1}.layer13);
layer6 = double(results{1,1}.layer14);
sum = double(results{1,1}.sum);
sumsq = double(results{1,1}.sumsq);
count = double(results{1,1}.count);
mean = sum/count;
a = sumsq/count-mean;
std = sqrt(max(a,0.01));

