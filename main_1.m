clc;
clear all;

 I = imread('1.jpg');
 %cut_Result=cut_patch(I,32);  %��ֵͼ����ȡ��Ե����Ե������ֵΪ1
 
 cat_Result=cat_patchs(I,32);