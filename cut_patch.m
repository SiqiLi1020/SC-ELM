function [patch_img] = cut_patch(RGB,patch_size)

% This function is to perform white balance for each RGB image and convert it to gray. 
% Then outputs image patches via setting patch size.

% input: RGB: a RGB image (H x W x C) H: height, W: width, C: channel;
%        patch_size: each iamge patch size
% output: patch_img: output image patches (H' x W' x N) H': patch
%         height, W': patch width, N: patch number. 
         

%% White balance; this is an optional operation
R = RGB(:,:,1);      G = RGB(:,:,2);      B = RGB(:,:,3);  
Rx4 = RGB(:,:,1)*4;  Gx4 = RGB(:,:,2)*4;  Bx4 = RGB(:,:,3)*4;  

Rave = mean(mean(R));   
Gave = mean(mean(G));   
Bave = mean(mean(B));  
Kave = (Rave + Gave + Bave) / 3;  

R1 = (Kave/Rave)*R; G1 = (Kave/Gave)*G; B1 = (Kave/Bave)*B;   
R2 = (Kave/Rave)*Rx4; G2 = (Kave/Gave)*Gx4; B2 = (Kave/Bave)*Bx4;   

RGB_white = cat(3, R1, G1, B1);  
RGB_whitex4 = cat(3, R2, G2, B2);  

RGB_white_out = uint8(RGB_white); RGB_white_outx4 = uint8(RGB_whitex4); 

%% Convert RGB to gray
img_gray = rgb2gray(RGB_white_out);

%% Image block
[m, n] = size(img_gray);
num_rowP = fix(m/patch_size);
num_colP = fix(n/patch_size);

%% Save the patch images to the specified folder
writeName = 'your store path'
writeName = 'H:\xyherself\5.1-SCI\博士学长666\picture-latex\18.4.26-论文小修\代码\cut_patch\';
index = 0;
for i = 1:num_rowP
   for j = 1: num_colP
       index = index + 1; 
       OutImg = img_gray(((i-1)*patch_size+1):i*patch_size , ((j-1)*patch_size+1):j*patch_size);
       patch_img(:,:,index) = OutImg;
       imwrite(OutImg,strcat([writeName,num2str(index),'.bmp']));
   end
end
end

