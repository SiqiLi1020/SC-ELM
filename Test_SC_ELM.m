function [test_time,average_dice,average_jaccard,average_precision,average_recall] = Test_SC_ELM(test_data,test_label,beneficial_weight)

% This function is to output predicted image patches and different evaluation indexes via SC-ELM-model and beneficial_weight.

% input: train_data: all testing data (x * y * z, x,y: patch size, x=y, z is the patch number);
%        train_label: all testing labeled images; (x * y * z);
%        beneficial_weight: beneficial weight set;
% output: test_time: testing time;
%         average_dice: average dice;
%         average_jaccard: average jaccard;
%         average_precision: average precision;
%         average_recall: average recall;
%         save predicted image patches;

tic;
%% load SC-ELM-model;
load SC-ELM-model.mat;

%% Data normalization [-1,1] or [-1,1]
[x, y, z] = size(test_data);
mapsize = x - KernelSize + 1;
for m = 1:z
    % [-1.1]
    test_data(:,:,m) = 2*(test_data(:,:,m)-min(min(test_data(:,:,m))))./(max(max(test_data(:,:,m)))-min(min(test_data(:,:,m))))-1;
    % [0,1]
    % train_data(:,:,m) = (train_data(:,:,m)-min(min(train_data(:,:,m))))./(max(max(train_data(:,:,m)))-min(min(train_data(:,:,m))));
end

%% Obtain feature maps based on same kernels and bias;
FeatureMap = cell(z, NumberofHiddenNeurons);
for i = 1:z
    for j = 1:NumberofHiddenNeurons
        FeatureMap{i,j} = convn(test_data(:,:,i), KernelData(:,:,j), 'valid') + down_Bias(:,:,j);
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

%% Obtain upsample feature maps based on same kernels' and new bias;
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

%% output predicted image patches via sum_UpsampleMap and beneficial_weight;
% method 1: first map by sigmoid and then compute the average
% all_outputs = cell(z,index_number);
% for i = 1:z
%     for j = 1:index_number
%         all_outputs{i,j} = sum_UpsampleMap{i,1} * beneficial_weight{j,1};
%         all_outputs{i,j} = sigm(all_outputs{i,j});
%     end
% end
% Final_output = cell(z,1);
% for i = 1:z
%     Final_output{i,1} = zeros(x,y);
% end
% for i = 1:z
%     for j = 1:index_number
%         Final_output{i,1} = Final_output{i,1} + all_outputs{i,j};
%     end
%     Final_output{i,1} = Final_output{i,1}./index_number;
% end
% for m = 1:z
%     for i = 1:x
%         for j = 1:y
%             if Final_output{m,1}(i,j) >= 1/2
%                 Final_output{m,1}(i,j) = 1;
%             else Final_output{m,1}(i,j) = 0;
%             end
%         end
%     end
% end

% method 2: first compute the average and then may by sigmoid;
all_outputs = cell(z,index_number);
for i = 1:z
    for j = 1:index_number
        all_outputs{i,j} = sum_UpsampleMap{i,1} * beneficial_weight{j,1};
    end
end
Final_output = cell(z,1);
for i = 1:z
    Final_output{i,1} = zeros(x,y);
end
for i = 1:z
    for j = 1:index_number
        Final_output{i,1} = Final_output{i,1} + all_outputs{i,j};
    end
    Final_output{i,1} = sigm(Final_output{i,1}./index_number);
end
for m = 1:z
    for i = 1:x
        for j = 1:y
            if Final_output{m,1}(i,j) >= 1/2
                Final_output{m,1}(i,j) = 1;
            else Final_output{m,1}(i,j) = 0;
            end
        end
    end
end


%% calculate average evaluation indexes
dice = zeros(z,1);
jaccard = zeros(z,1);
precision = zeros(z,1);
recall = zeros(z,1);
for i = 1:z
    dice(i,1) = Dice_Ratio(Final_output{i,1},test_label(:,:,i));
    jaccard(i,1) = Jaccard_Index(Final_output{i,1},test_label(:,:,i));
    precision(i,1) = Precision(Final_output{i,1},test_label(:,:,i));
    recall(i,1) = Recall(Final_output{i,1},test_label(:,:,i));
end

 average_dice = mean(dice);
 average_jaccard = mean(jaccard);
 average_precision = mean(precision);
 average_recall = mean(recall);

%% save predicted image patches
save ('predicted_patches', 'Final_output');

%% write predicted image patches to specified folder
% num = 0;
% for i = 1:z
%    	temp = Final_output{i,1};
%    	name = strcat ('your folder path', num2str(num), '.bmp');
%    	imwrite(temp, name, 'bmp');
%    	num = num + 1;
% end
test_time = toc;

end

