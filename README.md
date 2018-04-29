# SC-ELM：Structure Convolutional Extreme Learning Machine
 ## Definition
   
   SC-ELM is a project working for the use of segmentation.Please make sure you have the following configures for better use.
   
   · Different from the conventional extreme learning machine (ELM), SC-ELM can take an image as input and output a probability map directly with the same resolution of the input image. 
   
   · Meanwhile, following CNN, the convolution and de-convolution operations are utilized to extract and refine boundary and corner features. 
   
   · A large number of output weight matrices are learned by a single-hidden layer feedforward network (SLFN) and then a cross-feedback method is developed to select beneficial weight matrices for accurate segmentation.
 ## Functions
     
   Cut_patch: perform white balance for each RGB image and convert it to gray. <br>　　　　　 Then outputs image patches via setting patch size.
      
   Cat_patches: spell image patches into a gray image with original size. 
     

 ## Contact us
     
   Comments and criticism are greatly welcomed
     
   E-mail: sy.neu.lsq@gmail.com
