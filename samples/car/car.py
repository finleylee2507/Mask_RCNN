
import os
import sys
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import importlib
from mrcnn import visualize
importlib.reload(visualize)
from mrcnn import model as modellib, utils
from mrcnn.config import Config
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
import time
import datetime 
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import cv2
import matplotlib.pyplot as plt
import zipfile
import urllib.request
import shutil


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")



# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATA_PATH = os.path.join(ROOT_DIR,"samples/car/train/bdd100k/images/100k")
DEFAULT_MODEL_DIR = COCO_MODEL_PATH


############################################################
#  Configurations
############################################################


class CocoConfig(Config):
	"""Configuration for training on MS COCO.
	Derives from the base Config class and overrides values specific
	to the COCO dataset.
	"""
	# Give the configuration a recognizable name
	NAME = "coco"

	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 1

	GPU_COUNT = 1

	# Number of classes (including background)
	NUM_CLASSES = 1 + 11  # Deep drive has 10 classes


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
	def load_coco(self, dataset_dir, subset, class_ids=None,
				class_map=None, return_coco=False, auto_download=False):
		"""Load a subset of the COCO dataset.
		dataset_dir: The root directory of the COCO dataset.
		subset: What to load (train, val, minival, valminusminival)
		class_ids: If provided, only loads images that have the given classes.
		class_map: TODO: Not implemented yet. Supports maping classes from
			different datasets to the same class ID.
		return_coco: If True, returns the COCO object.
		auto_download: Automatically download and unzip MS-COCO images and annotations
		"""

		if auto_download is True:
			self.auto_download(dataset_dir, subset)

		if (subset == "train"):
			img_label = os.path.join(ROOT_DIR, "samples/car/train/bdd100k/images/100k/annotations/bdd100k_labels_images_det_coco_train.json")
		else:
			img_label = os.path.join(ROOT_DIR,"samples/car/train/bdd100k/images/100k/annotations/bdd100k_labels_images_det_coco_val.json")

		coco = COCO(img_label)  # bdd100k_labels_images_det_coco_train.json")
		# ("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
		print("Dataset dir: ", dataset_dir)
		if subset == "minival" or subset == "valminusminival": #select the validation directory 
			subset = "val"
		print("Subdirectory: ", subset)
		image_dir = "{}/{}".format(dataset_dir, subset)

		# Load all classes or a subset?
		if not class_ids:
			# All classes
			class_ids = sorted(coco.getCatIds())

		# All images or a subset?
		if class_ids:
			image_ids = []
			for id in class_ids:
				image_ids.extend(list(coco.getImgIds(catIds=[id])))
			# Remove duplicates
			image_ids = list(set(image_ids))
		else:
			# All images
			image_ids = list(coco.imgs.keys())

		# Add classes
		for i in class_ids:
			self.add_class("coco", i, coco.loadCats(i)[0]["name"])

		# Add images
		for i in image_ids:
			self.add_image(
				"coco", image_id=i,
				path=os.path.join(image_dir, coco.imgs[i]['file_name']),
				width=coco.imgs[i]["width"],
				height=coco.imgs[i]["height"],
				annotations=coco.loadAnns(coco.getAnnIds(
					imgIds=[i], catIds=class_ids, iscrowd=None)))
		if return_coco:
			return coco

	def load_mask(self, image_id):
		"""Load instance masks for the given image.

		Different datasets use different ways to store masks. This
		function converts the different mask format to one format
		in the form of a bitmap [height, width, instances].

		Returns:
		masks: A bool array of shape [height, width, instance count] with
			one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
		# If not a COCO image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "coco":
			return super(CocoDataset, self).load_mask(image_id)

		instance_masks = []
		class_ids = []
		annotations = self.image_info[image_id]["annotations"]
		# Build mask of shape [height, width, instance_count] and list
		# of class IDs that correspond to each channel of the mask.
		for annotation in annotations:
			class_id = self.map_source_class_id(
				"coco.{}".format(annotation['category_id']))
			if class_id:
				m = self.annToMask(annotation, image_info["height"],
								image_info["width"])
				# Some objects are so small that they're less than 1 pixel area
				# and end up rounded out. Skip those objects.
				if m.max() < 1:
					continue
				# Is it a crowd? If so, use a negative class ID.
				if annotation['iscrowd']:
					# Use negative class ID for crowds
					class_id *= -1
					# For crowd masks, annToMask() sometimes returns a mask
					# smaller than the given dimensions. If so, resize it.
					if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
						m = np.ones(
							[image_info["height"], image_info["width"]], dtype=bool)
				instance_masks.append(m)
				class_ids.append(class_id)

		# Pack instance masks into an array
		if class_ids:
			mask = np.stack(instance_masks, axis=2).astype(np.bool)
			class_ids = np.array(class_ids, dtype=np.int32)
			return mask, class_ids
		else:
			# Call super class to return an empty mask
			return super(CocoDataset, self).load_mask(image_id)

	def image_reference(self, image_id): #not really useful in this case 
		print("Not applicable to the bdd100k dataset")
		# """Return a link to the image in the COCO Website."""
		# info = self.image_info[image_id]
		# if info["source"] == "coco":
		# 	return "http://cocodataset.org/#explore?id={}".format(info["id"])
		# else:
		# 	super(CocoDataset, self).image_reference(image_id)

	# The following two functions are from pycocotools with a few changes.

	def annToRLE(self, ann, height, width):
		"""
		Convert annotation which can be polygons, uncompressed RLE to RLE.
		:return: binary mask (numpy 2D array)
		"""
		segm = ann['segmentation']
		if isinstance(segm, list):
			# polygon -- a single object might consist of multiple parts
			# we merge all parts into one mask rle code
			rles = maskUtils.frPyObjects(segm, height, width)
			rle = maskUtils.merge(rles)
		elif isinstance(segm['counts'], list):
			# uncompressed RLE
			rle = maskUtils.frPyObjects(segm, height, width)
		else:
			# rle
			rle = ann['segmentation']
		return rle

	def annToMask(self, ann, height, width):
		"""
		Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
		:return: binary mask (numpy 2D array)
		"""
		rle = self.annToRLE(ann, height, width)
		m = maskUtils.decode(rle)
		return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
	"""Arrange resutls to match COCO specs in http://cocodataset.org/#format
	"""
	# If no results, return an empty list
	if rois is None:
		return []

	results = []
	for image_id in image_ids:
		# Loop through detections
		for i in range(rois.shape[0]):
			class_id = class_ids[i]
			score = scores[i]
			bbox = np.around(rois[i], 1)
			mask = masks[:, :, i]

			result = {
				"image_id": image_id,
				"category_id": dataset.get_source_class_id(class_id, "coco"),
				"bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
				"score": score,
				"segmentation": maskUtils.encode(np.asfortranarray(mask))
			}
			results.append(result)
	return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
	"""Runs official COCO evaluation.
	dataset: A Dataset object with valiadtion data
	eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
	limit: if not 0, it's the number of images to use for evaluation
	"""
	# Pick COCO images from the dataset
	image_ids = image_ids or dataset.image_ids

	# Limit to a subset
	if limit:
		image_ids = image_ids[:limit]

	# Get corresponding COCO image IDs.
	coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

	t_prediction = 0
	t_start = time.time()

	results = []
	for i, image_id in enumerate(image_ids):
		# Load image
		image = dataset.load_image(image_id)

		# Run detection
		t = time.time()
		r = model.detect([image], verbose=0)[0]
		t_prediction += (time.time() - t)

		# Convert results to COCO format
		# Cast masks to uint8 because COCO tools errors out on bool
		image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
										r["rois"], r["class_ids"],
										r["scores"],
										r["masks"].astype(np.uint8))
		results.extend(image_results)

	# Load results. This modifies results with additional attributes.
	coco_results = coco.loadRes(results)

	# Evaluate
	cocoEval = COCOeval(coco, coco_results, eval_type)
	cocoEval.params.imgIds = coco_image_ids
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()

	print("Prediction time: {}. Average {}/image".format(
		t_prediction, t_prediction / len(image_ids)))
	print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
	import argparse

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train Mask R-CNN on BDD')
	parser.add_argument("command",
						metavar="<command>",
						help="'train', 'evaluate' or 'detect' on BDD")
	parser.add_argument('--dataset', required=False,
						default=DEFAULT_DATA_PATH,
						metavar=DEFAULT_DATA_PATH,
						help='Directory of the BDD dataset')
	parser.add_argument('--model', required=False,
						default=DEFAULT_MODEL_DIR,
						metavar=DEFAULT_MODEL_DIR,
						help="Path to weights .h5 file")
	parser.add_argument('--logs', required=False,
						default=DEFAULT_LOGS_DIR,
						metavar="/home/dtilak/object_detection/training_log_rchannel/",
						help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--limit', required=False,
						default=500,
						metavar="<image count>",
						help='Images to use for evaluation (default=500)') 
	parser.add_argument('--imgpath',required=False, help='Path to the image for detection')
	parser.add_argument('--videopath',required=False,help='Path to the video for detection')
	parser.add_argument('--maskoff',required=False,default=False,action='store_true')
	parser.add_argument('--write',required=False,default=False,action='store_true')
	parser.add_argument('--thre',required=False, help='Threshold for detection result',default=0.8, action="store" )

	#path to save the detection result
	save_path='detection_result'

	args = parser.parse_args()
	print("Command: ", args.command)
	print("Model: ", args.model)
	print("Dataset: ", args.dataset)
	print("Logs: ", args.logs)
	print("Image Path: ",args.imgpath)

	# Configurations
	if args.command == "train":
		config = CocoConfig()
	else:
		class InferenceConfig(CocoConfig):  # subclass of original config
			# Set batch size to 1 since we'll be running inference on
			# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
			GPU_COUNT = 1
			IMAGES_PER_GPU = 1
			DETECTION_MIN_CONFIDENCE = 0
		config = InferenceConfig()
	config.display()

	# Create model
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,
								model_dir=args.logs)
	else:
		model = modellib.MaskRCNN(mode="inference", config=config,
								model_dir=args.logs)

	# Select weights file to load
	if args.model.lower() == "coco":
		model_path = COCO_MODEL_PATH
	elif args.model.lower() == "last":
		# Find last trained weights
		model_path = model.find_last()
	elif args.model.lower() == "imagenet":
		# Start from ImageNet trained weights
		model_path = model.get_imagenet_weights()
	else:
		model_path = args.model

	# Load weights
	print("Loading weights ", model_path)
	# Exclude the last layers because they require a matching number of classes
	if (args.command =="train"):
		model.load_weights(model_path, by_name=True, exclude=[
					"mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
	else:
		model.load_weights(model_path, by_name=True)


	

	# Train or evaluate
	if args.command == "train":
		# Training dataset. Use the training set and 35K from the
		# validation set, as as in the Mask RCNN paper.
		dataset_train = CocoDataset()
		# , year=args.year, auto_download=args.download)
		dataset_train.load_coco(args.dataset, "train")
		# if args.year in '2014':
		#    dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
		dataset_train.prepare()

		# Validation dataset
		dataset_val = CocoDataset()
		val_type = "val"  # if args.year in '2017' else "minival"
		# , year=args.year, auto_download=args.download)
		dataset_val.load_coco(args.dataset, val_type)
		dataset_val.prepare()

		# Image Augmentation
		# Right/Left flip 50% of the time
		augmentation = imgaug.augmenters.Fliplr(0.5)

		# *** This training schedule is an example. Update to your needs ***

		# Training - Stage 1
		print("Training network heads")
		model.train(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE,
					epochs=5,
					layers='heads',
					augmentation=augmentation)

		# Training - Stage 2
		# Finetune layers from ResNet stage 4 and up
		print("Fine tune Resnet stage 4 and up")
		model.train(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE,
					epochs=5,
					layers='4+',
					augmentation=augmentation)

		# Training - Stage 3
		# Fine tune all layers
		print("Fine tune all layers")
		model.train(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE / 10,
					epochs=5,
					layers='all',
					augmentation=augmentation)

	elif args.command == "evaluate":
		# Validation dataset
		dataset_val = CocoDataset()
		val_type = "val"  # if args.year in '2017' else "minival"
		# ,year=args.year, auto_download=args.download)
		coco = dataset_val.load_coco(args.dataset, val_type,  return_coco=True)
		dataset_val.prepare()
		print("Running COCO evaluation on {} images.".format(args.limit))
		evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
	elif args.command =="detect":
		#run detection on the input image
		if(args.imgpath):
			dataset_detect=CocoDataset()
			val_type="minival" 
			dataset_detect.load_coco(args.dataset,val_type) 
			dataset_detect.prepare()
			image_path=args.imgpath
			print(image_path)
			image=cv2.imread(image_path)
			fig, ax = plt.subplots(1, figsize=(26, 26))
			height, width = image.shape[:2]

			
			
			results = model.detect([image], verbose=1)

			#display result 
			
			
			r=results[0]
			#print(r)
			#filter out weak detection results 
			thre=float(args.thre) 
			temp=dict()
			selected=r['scores']>thre #index of the elements with prob > threshold 
			new_rois=r['rois'][selected]
			new_masks=r['masks'][:,:,selected]
			new_classids=r['class_ids'][selected]
			new_scores=r['scores'][selected]
			temp['rois']=new_rois 
			temp['masks']=new_masks
			temp['class_ids']=new_classids
			temp['scores']=new_scores 

        		




			if(not args.maskoff): #if displayed mask option is on 
				image_ir = visualize.display_instances(image, temp['rois'], temp['masks'],temp['class_ids'], 
									dataset_detect.class_names, temp['scores'],ax=ax, 
									title="Predictions")
				# image_ir = visualize.display_instances_new(image, r['rois'], r['masks'],r['class_ids'], 
				# 					dataset_detect.class_names, r['scores'],ax=ax, 
				# 					)
				# plt.imshow(image_ir)
				# plt.show()
				#write the detection result to a folder 
			   
				if(args.write):
					plt.savefig(os.path.join(save_path,os.path.split(image_path)[1]),bbox_inches='tight')
					
				plt.show()
			else:
				captions=["{} {:.3f}".format(dataset_detect.class_names[int(c)], s) if c > 0 else ""
				for c, s in zip(temp['class_ids'], temp['scores'])]

				visualize.draw_boxes(
				image, 
				refined_boxes=temp['rois'],
				captions=captions, title="Detections",ax=ax) 
				#write the detection result to a folder 
		
				if(args.write):
					plt.savefig(os.path.join(save_path,os.path.split(image_path)[1]),bbox_inches='tight')
				
				plt.show()


		#run detection on the input video 
		if(args.videopath):
			dataset_detect=CocoDataset()
			val_type="minival" 
			dataset_detect.load_coco(args.dataset,val_type) 
			dataset_detect.prepare()
			video_path=args.videopath
			vcapture = cv2.VideoCapture(video_path)
			width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps = vcapture.get(cv2.CAP_PROP_FPS)
			fig, ax = plt.subplots(1, figsize=(26, 26))
			# Define codec and create video writer
			file_name = "detection_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
			vwriter = cv2.VideoWriter(file_name,
								  cv2.VideoWriter_fourcc(*'MJPG'),
								  fps, (width, height))
			
			count=0
			success=True 
			
			while success:
				print("frame: ", count)
				 # Read next image
				success, image = vcapture.read()
				if success:
					# OpenCV returns images as BGR, convert to RGB
					image = image[..., ::-1] #? how 
					
					# Detect objects
					r = model.detect([image], verbose=0)[0]

								
					
					#filter out weak detection results 
					thre=float(args.thre) 
					temp=dict()
					selected=r['scores']>thre #index of the elements with prob > threshold 
					new_rois=r['rois'][selected]
					new_masks=r['masks'][:,:,selected]
					new_classids=r['class_ids'][selected]
					new_scores=r['scores'][selected]
					temp['rois']=new_rois 
					temp['masks']=new_masks
					temp['class_ids']=new_classids
					temp['scores']=new_scores 
							

					
					image_ir=visualize.display_instances_new(image, temp['rois'], temp['masks'],temp['class_ids'], 
									dataset_detect.class_names, temp['scores']
									)
					
					

					
					# plt.savefig('temp.jpg',bbox_inches='tight')
					# image_ir=cv2.imread('temp.jpg')
					

					
					# RGB -> BGR to save image to video
					image_ir = image_ir[..., ::-1]
					cv2.imshow("Detection",image_ir)
					cv2.waitKey(0)
   					


					
					
					# plt.imshow(np.uint8(image_ir))
					# plt.show()
					
					# Add image to video writer
					vwriter.write(np.uint8(image_ir))
					count += 1 
			vwriter.release()
			print("Saved to ", file_name)

				
			

		
		
		

		
		


	else:
		print("'{}' is not recognized. "
			"Use 'train' or 'evaluate'".format(args.command))
