function [WSI_img] = cat_patchs(RGB, patch_size)
 
% This function is to spell image patches into a gray image with original size. 

% input: RGB: a RGB image (H * W * C) H: height, W: width, C: channel;
%        patch_size: each image patch size
% output: WSI_img: a gray image with original size (H * W) 

%% your store folder of image patches
%fileName = 'your store path'
fileName = 'H:\xyherself\5.1-SCI\博士学长666\picture-latex\18.4.26-论文小修\代码\cut_patch\';
img_gray =  rgb2gray(RGB);
[m, n] = size(img_gray);
num_rowP = fix(m/patch_size);
num_colP = fix(n/patch_size);


%% Take all the images into one column;
imgName = [fileName, num2str(1), '.bmp'];
I = imread(imgName);
WSI_line_img = I;
index = 0;
for i = 1:num_rowP
    for j = 1: num_colP
       index = index + 1; 
       imgName1 = [fileName, num2str(index), '.bmp'];
       I1 = imread(imgName1);
       WSI_line_img = [WSI_line_img I1];
    end
end
% noted that, this need your calculation that fit your image size, our
% image is 1280 * 960;
WSI_line_img = WSI_line_img(:,33:38432);

%% spell image;
WSI_img = zeros(1,1280);
WSI_img = uint8(WSI_img);

for j = 1:num_rowP
    rowline = WSI_line_img(:, num_colP*patch_size*(j-1)+1 : num_colP*patch_size*j);
    WSI_img = [WSI_img; rowline];
end
% noted that, this need your calculation that fit your image size
WSI_img = WSI_img(2:961, :);
end




