# YOLO9000_COCO
This is an object-detection network implemented with Tensorflow for YOLO9000. The dataset used to train the model is COCO dataset. 

In this implementation, some modification for loss function was made to improve the robutness of the model.   The loss of the model consists of the coordinate loss (xy coordinate loss, hw coordinate loss), the confidence loss (no objec loss, object loss), and the classification loss. In YOLO9000, the all kinds of loss are calculated using MSE. In this project, they are calculated with cross entropy, except hw coordinate loss. 

For other details, please refer to the  [paper](https://pjreddie.com/darknet/yolo/) and the [dataset](http://cocodataset.org/).

# Environment
- python 3.6
- TensorFlow 1.10
- opencv 2
- [pycocotools](https://pypi.org/project/pycocotools/2.0.0/) （COCO dataset api）

# How to Use
##### For Training
- Download the COCO dataset.
- Convert the images in the COCO dataset (training folder) to TFRecord by running the TFRecords.py.
- Change the **_istrain_** flag to be **_True_** in the **_ _main()_** function in the main.py file.
- Run!

##### For Testing
- Finish the 4 steps in the training. (If you have trained your model, skip this step.)
- Convert the images in the COCO dataset (testing folder) to TFRecord by running the TFRecords.py.
- Change the **_istrain_** flag to be **_False_** in the **_ _main()_** function in the main.py file.
- Run!

# What You Will See
### For Training
- The loss information.
- The examples of images predicted by the network will be saved into the _./images/_ folder.

##### Training for One Epoch:
<div align=center><img src="./demo_img/epoch1_step7200_i0.png" width="500px/"></div>

##### Training for Three Epochs:
<div align=center><img src="./demo_img/epoch3_step1700_i1.png" width="500px/"></div>


### For Testing
- The result of testing images will be saved into the _./images/_ floder.

##### Example Testing Images ():
One image.
<div align=center><img src="./demo_img/step24_i10.png" width="500px/"></div>

Annnd another one.
<div align=center><img src="./demo_img/step30_i14.png" width="500px/"></div>

Annnnnnnnd one more times!
<div align=center><img src="./demo_img/step41_i1.png" width="500px/"></div>

# For More
Contact me: vxallset@outlook.com
