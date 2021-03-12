# Object detection and segmentation on self-driving cars 
An object detection project on the bdd-100k dataset

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

![Instance Segmentation Sample](images/predictions.jpg) 

Note: This implementation is a modified and updated version of the following project: 
https://github.com/TilakD/Object-detection-and-segmentation-for-self-driving-cars 

It is also based on and inspired by the official Mask R-CNN impelmentation by Matterport: https://github.com/matterport/Mask_RCNN  


The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101.
* Soure codes for training on the BDD100K dataset 
* Pre-trained weights for BDD100K
* Jupyter notebooks to visualize the detection pipeline at every step
* Parallel Model class for multi-GPU training



# Getting Started
* [demo.ipynb](object_detection/demo.ipynb) Is the easiest way to start. It shows an example of using a model pre-trained on BDD100k to segment objects in your own images.
It includes code to run object detection and instance segmentation on arbitrary images.

* ([model.py](object_detection/mrcnn/model.py), [utils.py](object_detection/mrcnn/utils.py), [config.py](object_detection/mrcnn/config.py)): These files contain the main Mask RCNN implementation. 


* [inspect_data.ipynb](object_detection/inspect_data.ipynb). This notebook visualizes the different pre-processing steps
to prepare the training data.

* [inspect_model.ipynb](object_detection/inspect_model.ipynb) This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

* [inspect_weights.ipynb](object_detection/inspect_weights.ipynb)
This notebooks inspects the weights of a trained model and looks for anomalies and odd patterns.


# How to train the model on the BDD-100K dataset 
1. Donwload the bdd-100k dataset (https://github.com/ucbdrive/bdd-data)
2. Convert the downloaded training and testing labels into COCO formats using bdd2coco.py in the bdd-data-master folder. (citation: https://github.com/TilakD/Object-detection-and-segmentation-for-self-driving-cars) 
3. cd into the samples/car directory. Create a folder named "train" and place the downloaded dataset and the converted label inside the folder. 
4. Run car.py to train the model. 
5. After the training, you can run car.py to perform detection on images and videos. Check out some of the example commands you could use in commands.txt. 
6. Enjoy! 

