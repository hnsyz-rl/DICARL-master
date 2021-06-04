%  Please complete the section using the "%@" logo before run the code.

%read data
clear;
[datapoints, numpoints] = readdata('w1.bin');
len = length(datapoints);
p_sp = datapoints(1, 1:len);  % m/s2
q_sp = datapoints(2, 1:len);
r_sp = datapoints(3, 1:len);
p = datapoints(4, 1:len);  % rad/s
q = datapoints(5, 1:len);
r = datapoints(6, 1:len);

% calcule the average error of the data
delta_p = sum(abs(p_sp-p))/length(p)
delta_q = sum(abs(q_sp-q))/length(q)
delta_r = sum(abs(results-r))/length(r)


%Draw the data into a linear graph
figure('color','w');
t = [0:0.001:(length(p)-1)/1000];
subplot(3,1,1);
plot(t, p_sp);
hold on
plot(t,p);
ylabel('Roll(deg/s)');
legend('p-期望','p');

subplot(3,1,2);
plot(t, q_sp);
hold on
plot(t,q);
legend('q-期望','q');
ylabel('Pitch(deg/s)');

subplot(3,1,3);
plot(t, results);
hold on
plot(t,r);
ylabel('Yaw(deg/s)');
legend('r-期望','r');
xlabel('time/s');





%% Load the data into MATLAB from a binary log file
% Usage: >> [datapoints, numpoints] = readdata('datafile.log')
% Header information format:
%           String "MWLOGV##"
%           Time/Date 4 bytes (time())
%           Number of Signals per record Logged 1 bytes (256 max)
%           Data Type of Signals Logged  1 bytes (1-10)
%           Number of bytes per record 2 (65535 max)
% Plot Data Example: plot([1:numpoints], datapoints(1,:), [1:numpoints], datapoints(2,:))
% MathWorks Pilot Engineering 2015
% Steve Kuznicki
function [datapts, numpts] = readdata(dataFile)
%%
datapts = 0;
numpts = 0;

if nargin == 0
    dataFile = 'data.bin';
end

fid = fopen(dataFile, 'r');
% load the header information
hdrToken = fread(fid, 8, 'char');
if strncmp(char(hdrToken),'MWLOGV',6) == true
    logTime = uint32(fread(fid, 1, 'uint32'));
    numflds = double(fread(fid, 1, 'uint8'));
    typefld = uint8(fread(fid, 1, 'uint8'));
    recSize = uint16(fread(fid, 1, 'uint16'));
    fieldTypeStr = get_elem_type(typefld);
    datapts = fread(fid, double([numflds, Inf]), fieldTypeStr);
    fclose(fid);
    numpts = size(datapts,2);
end

end

%% get the element type string
function [dtypeStr] = get_elem_type(dtype)
    switch(dtype)
        case 1
            dtypeStr = 'double';
        case 2
            dtypeStr = 'single';
        case 3
            dtypeStr = 'int32';
        case 4
            dtypeStr = 'uint32';
        case 5
            dtypeStr = 'int16';
        case 6
            dtypeStr = 'uint16';
        case 7
            dtypeStr = 'int8';
        case 8
            dtypeStr = 'uint8';
        case 9
            dtypeStr = 'logical';
        case 10
            dtypeStr = 'embedded.fi';
    end
end
