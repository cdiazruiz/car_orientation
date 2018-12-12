# car_orientation
Car Orientation Estimation from Monocular Image 

Download code for Mask RCNN from : https://github.com/matterport/Mask_RCNN
  Follow instrunctions on setup and requirements

Kitti evaluation code:
  https://github.com/prclibo/kitti_eval
  
  
Save the resnet file within the samples documents within the mask rcnn repo:
hyperparameter_random.py to run random search hyperparameter optimization.
  In the file you need to pass a training and validation set with form:
    
    trainingsplit
    
      -000000_0.png
      
      -000000_1.png
      
      -orientation_label.txt
      
     valsplit
     
      -000000_0.png
      
      -000000_1.png
      
      -orientation_label.txt
      


