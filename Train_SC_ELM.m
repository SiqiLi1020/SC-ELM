function [train_time, beneficial_weight] = Train_SC_ELM(train_data,train_label,NumberofHiddenNeurons,KernelSize,index_number)

% This function is to train SC-ELM and output the SC-ELM-model and beneficial weight set.

% input: train_data: all training data (x * y * z, x,y: patch size, x=y, z is the patch number);
%        train_label: all training labeled images (x * y * z);
%        NumberofHiddenNeurons: number of hidden neurons;
%        KernelSize: size of kernel;
%        index_number: the number of the most beneficial output weights;
% output: train_time: trainging time;
%         beneficial_weight: beneficial weight set;
%         save SC-ELM-model: include NumberofHiddenNeurons, KernelSize, KernelData, down_Bias, up_Bias and output_weight

tic;
%% Data normalization [-1,1] or [-1,1]
[x, y, z] = size(train_data);
for m = 1:z
    % [-1.1]
    train_data(:,:,m) = 2*(train_data(:,:,m)-min(min(train_data(:,:,m))))./(max(max(train_data(:,:,m)))-min(min(train_data(:,:,m))))-1;
    % [0,1]
    % train_data(:,:,m) = (train_data(:,:,m)-min(min(train_data(:,:,m))))./(max(max(train_data(:,:,m)))-min(min(train_data(:,:,m))));
end

%% Initialize convolution kernels and bias randomly;
% NumberofHiddenNeurons = 64; % number of hidden neurons
% KernelSize = 11; % size of kernel
mapsize = x - KernelSize + 1;
KernelData = zeros(KernelSize,KernelSize,NumberofHiddenNeurons);
down_Bias = zeros(mapsize,mapsize,NumberofHiddenNeurons);
for m = 1:NumberofHiddenNeurons
    KernelData(:,:,m) = 2*rand(KernelSize) - 1;
    down_Bias(:,:,m) = rand(mapsize,mapsize);
end

%% Obtain feature maps based on kernels and bias;
FeatureMap = cell(z, NumberofHiddenNeurons);
for i = 1:z
    for j = 1:NumberofHiddenNeurons
        FeatureMap{i,j} = convn(train_data(:,:,i), KernelData(:,:,j), 'valid') + down_Bias(:,:,j);
    end
end

%% sum feature maps and map via different activation functions;
sum_FeatureMap = cell(z,1);
for i = 1:z
    sum_FeatureMap{i,1} = zeros(mapsize,mapsize);
end
for i = 1:z
    for j = 1:NumberofHiddenNeurons
        sum_FeatureMap{i,1} = sum_FeatureMap{i,1} + FeatureMap{i,j};
        % map by tanh;
        sum_FeatureMap{i,1} = tanh(sum_FeatureMap{i,1});
        % map by sigmoid;
        %sum_FeatureMap{i,1} = sigm(sum_FeatureMap{i,1});
    end
end

%% Obtain upsample feature maps based on kernels' and new bias;
up_Bias = zeros(x,y,NumberofHiddenNeurons);
for m = 1:NumberofHiddenNeurons
    up_Bias(:,:,m) = rand(x,y);
end
UpsampleMap = cell(z, NumberofHiddenNeurons);
for i = 1:z
    for j = 1:NumberofHiddenNeurons
        UpsampleMap{i,j} = convn(sum_FeatureMap{i,1}, rot180(KernelData(:,:,j)), 'full') + up_Bias(:,:,j);
    end
end

%% sum upsample feature maps and map via different activation functions (note that, you can combine different activation functions and need no one-to-one correspondence);
sum_UpsampleMap = cell(z,1);
for i = 1:z
    sum_UpsampleMap{i,1} = zeros(x,y);
end
for i = 1:z
    for j = 1:NumberofHiddenNeurons
        sum_UpsampleMap{i,1} = sum_UpsampleMap{i,1} + UpsampleMap{i,j};
        % map by tanh;
        sum_UpsampleMap{i,1} = tanh(sum_UpsampleMap{i,1});
        % map by sigmoid (Although you choose tanh function as the activation function on sum_FeatureMap);
        %sum_UpsampleMap{i,1} = sigm(sum_UpsampleMap{i,1});
    end
end
%% obtain output weight;
output_weight = cell(z,1);
for i = 1:z
    output_weight{i,1} = pinv(sum_UpsampleMap{i,1}) * train_label(:,:,i);
end

%% obtain beneficial output weight matrix
beneficial_weight = optimize_weight(sum_UpsampleMap,output_weight,train_label,index_number);

%% training time
train_time = toc;

%% save SC-ELM model
save('SC-ELM-model','NumberofHiddenNeurons','KernelSize','KernelData','down_Bias','up_Bias','output_weight','index_number');

end


