function [beneficial_weight] = optimize_weight(sum_UpsampleMap,output_weight,train_label,index_number)

% This function is to optimize output weights. 

% input: sum_UpsampleMap: all training upsample maps;
%        output_weight: all obtained output weights;
%        train_label: all training labeled images;
%        index_number: the number of the most beneficial output weights;
% output: beneficial_weight: beneficial weight set


[x,y,z] = size(train_label);

%% mixing matrix using all training upsample maps and obtained output weights;
mix_mat = cell(z,z);
for i = 1:z
    for j = 1:z
        mix_mat{i,j} = sum_UpsampleMap{i,1} * output_weight{j,1};
    end
end

for i = 1:z
    for j = 1:z
        for x1 = 1:x
            for y1 = 1:y
                if mix_mat{i,j}(x1,y1) >= 0;
                    mix_mat{i,j}(x1,y1) = 1;
                else mix_mat{i,j}(x1,y1) = 0;
                end
            end
        end
    end
end

%% calculate the score of each output weight and rank from high to low via scores;
weightscore_mat = zeros(z,z);
for i = 1:z
    for j = 1:z
        if Dice_Ratio(mix_mat{i,j},train_label(:,:,i)) >= 0.7;
            weightscore_mat(i,j) = 1;
        end
    end
end
weight_score = zeros(1,z);
for i = 1:z
    weight_score(1,i) = sum(weightscore_mat(i,:));
end
[rank,index]=sort(weight_score,'descend');

%% obtain beneficial weight set via index_number;
beneficial_weight = cell(index_number,1);
for i = 1:index_number
    beneficial_weight{i,1} = output_weight{index(1,i)};
end

end



