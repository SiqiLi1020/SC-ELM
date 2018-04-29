%% this is a demo to train and test ELM;

%% step 1. cut image into patches;
% e.g., [patch_img] = cut_patch(RGB,patch_size);

%% step 2. train SC-ELM;
% load your training patches and labeled images, set NumberofHiddenNeurons,KernelSiz and index_number
% e.g.,[train_time, beneficial_weight] = Train_SC_ELM(train_data,train_label,NumberofHiddenNeurons,KernelSize,index_number)

%% step 3. test SC-ELM and output predicted patches
% load your testing patches, test labeled images and beneficial weight set
% e.g., [test_time,average_dice,average_jaccard,average_precision,average_recall] = Test_SC_ELM(test_data,test_label,beneficial_weight)

%% step 4. spell image patches into the gray image with original size. 
% load your path of  predicted patches;
% e.g., [WSI_img] = cat_patchs(RGB, patch_size)

%% WSI_img is the Global segmentation result