%  Please complete the section using the "%@" logo before run the code.
clc;
clear;
figure(1);


fid=fopen('DICARL.bin', "r");
[DICARL_datapoints, NN_numpoints] = fread(fid, [6, Inf], 'double');

fid=fopen('PID.bin', "r");
[PID_datapoints, PID_numpoints] = fread(fid, [6, Inf], 'double');


NN_p_sp = DICARL_datapoints(1,:);  % m/s2
NN_q_sp = DICARL_datapoints(2,:);
NN_r_sp = DICARL_datapoints(3,:);
NN_p = DICARL_datapoints(4,:);  % rad/s
NN_q = DICARL_datapoints(5,:);
NN_r = DICARL_datapoints(6,:);

NN_p = NN_p/max(abs(NN_p_sp));
NN_p_sp = NN_p_sp/max(abs(NN_p_sp));

NN_q = NN_q/max(abs(NN_q_sp));
NN_q_sp = NN_q_sp/max(abs(NN_q_sp));

NN_r = NN_r/max(abs(NN_r_sp));
NN_r_sp = NN_r_sp/max(abs(NN_r_sp));

NN_delta_p = sum(abs(NN_p_sp-NN_p))/length(NN_p);
NN_delta_q = sum(abs(NN_q_sp-NN_q))/length(NN_q);
NN_delta_r = sum(abs(NN_r_sp-NN_r))/length(NN_r);

t = [0:0.01:(length(NN_p)-1)/100];
legend_fontsize = 12;
title_fontsize = 28;
alpha = 0.7;
subplot(3,1,1);
plot(t, NN_p_sp, '--k', 'LineWidth',1,'MarkerSize',8);
hold on
actual = plot(t,NN_p, 'r', 'LineWidth',1,'MarkerSize',8);
actual.Color(4) = alpha;
legend('Desired','Actual');
ylabel('Roll(deg/s)');

title('DICARL controller','FontName','Times New Roman','FontSize',title_fontsize);
subplot(3,1,2);
plot(t, NN_q_sp, '--k', 'LineWidth',1,'MarkerSize',8);
hold on
actual = plot(t,NN_q, 'r', 'LineWidth',1,'MarkerSize',8);
actual.Color(4) = alpha;
ylabel('Pitch(deg/s)');
legend('Desired','Actual');

subplot(3,1,3);
plot(t, NN_r_sp, '--k', 'LineWidth',1,'MarkerSize',8);
hold on
actual = plot(t,NN_r, 'r', 'LineWidth',1,'MarkerSize',8);
actual.Color(4) = alpha;
legend('Desired','Actual');
ylabel('Yaw(deg/s)');

xlabel('time/s');
set(gcf,'color' ,'w');
set(gca,'GridLineStyle' ,'-');     


figure(2);

alpha = 0.4;

PID_p_sp = PID_datapoints(1,:);  % m/s2
PID_q_sp = PID_datapoints(2,:);
PID_r_sp = PID_datapoints(3,:);
PID_p = PID_datapoints(4,:);  % rad/s
PID_q = PID_datapoints(5,:);
PID_r = PID_datapoints(6,:);


PID_p = PID_p/max(abs(PID_p_sp));
PID_p_sp = PID_p_sp/max(abs(PID_p_sp));

PID_q = PID_q/max(abs(PID_q_sp));
PID_q_sp = PID_q_sp/max(abs(PID_q_sp));

PID_r = PID_r/max(abs(PID_r_sp));
PID_r_sp = PID_r_sp/max(abs(PID_r_sp));

PID_delta_p = sum(abs(PID_p_sp-PID_p))/length(PID_p);
PID_delta_q = sum(abs(PID_q_sp-PID_q))/length(PID_q);
PID_delta_r = sum(abs(PID_r_sp-PID_r))/length(PID_r);

t = [0:0.01:(length(PID_p)-1)/100];

subplot(3,1,1);
plot(t, PID_p_sp, '--k', 'LineWidth',1,'MarkerSize',8);
hold on
actual = plot(t,PID_p, 'r', 'LineWidth',1,'MarkerSize',8);
actual.Color(4) = alpha;
ylabel('Roll(deg/s)');
legend('Desired','Actual');

title('PID controller','FontName','Times New Roman','FontSize',title_fontsize);
subplot(3,1,2);
plot(t, PID_q_sp, '--k', 'LineWidth',1,'MarkerSize',8);
hold on
actual = plot(t,PID_q, 'r', 'LineWidth',1,'MarkerSize',8);
actual.Color(4) = alpha;
legend('Desired','Actual');
ylabel('Pitch(deg/s)');

subplot(3,1,3);
plot(t, PID_r_sp, '--k', 'LineWidth',1,'MarkerSize',8);
hold on
actual = plot(t,PID_r, 'r', 'LineWidth',1,'MarkerSize',8);
actual.Color(4) = alpha;
ylabel('Yaw(deg/s)');
legend('Desired','Actual');
xlabel('time/s');

set(gcf,'color' ,'w');
set(gca,'GridLineStyle' ,'-');     

PID_mean_p = mean(abs(PID_p_sp-PID_p))
PID_std_p = std(abs(PID_p_sp-PID_p));

PID_mean_q = mean(abs(PID_q_sp-PID_q))
PID_std_q = std(abs(PID_q_sp-PID_q));

PID_mean_r = mean(abs(PID_r_sp-PID_r))
PID_std_r = std(abs(PID_r_sp-PID_r));

NN_mean_p = mean(abs(NN_p_sp-NN_p))
NN_std_p = std(abs(NN_p_sp-NN_p));

NN_mean_q = mean(abs(NN_q_sp-NN_q))
NN_std_q = std(abs(NN_q_sp-NN_q));

NN_mean_r = mean(abs(NN_r_sp-NN_r))
NN_std_r = std(abs(NN_r_sp-NN_r));
