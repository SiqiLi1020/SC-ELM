clc;
clear all;

 I = imread('1.jpg');
 %cut_Result=cut_patch(I,32);  %二值图像提取边缘，边缘的像素值为1
 
 cat_Result=cat_patchs(I,32);