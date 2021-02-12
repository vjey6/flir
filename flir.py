"""
Mask R-CNN
Train on the toy bottle dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 bottle.py train --dataset=/home/datascience/Workspace/maskRcnn/Mask_RCNN-master/samples/bottle/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 bottle.py train --dataset=/path/to/bottle/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 bottle.py train --dataset=/path/to/bottle/dataset --weights=imagenet
    # Apply color splash to an image
    python3 bottle.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 bottle.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/drive/MyDrive/flir")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
  """Configuration for training on the toy  dataset.
  Derives from the base Config class and overrides some values.
  """
  # Give the configuration a recognizable name
  NAME = "object"

  # We use a GPU with 12GB memory, which can fit two images.
  # Adjust down if you use a smaller GPU.
  IMAGES_PER_GPU = 2

  # Number of classes (including background)
  NUM_CLASSES = 1 + 4  # Background + toy

  # Number of training steps per epoch
  STEPS_PER_EPOCH = 100

  # Skip detections with < 90% confidence
  DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):
 def load_custom(self,dataset_dir, subset):
   #Add classes as per your requirement and order
   self.add_class('object', 1, 'Person')
   self.add_class('object', 2, 'Car')
   self.add_class('object', 3, 'Dog')
   self.add_class('object', 4, 'Chair')
   assert subset in ['train', 'val']
   dataset_dir = os.path.join(dataset_dir, subset)
   annotations = json.load(open(os.path.join(dataset_dir,
                                 'via_region_data.json')))
   annotations = list(annotations.values()) 
   annotations = [a for a in annotations if a['regions']]
   for a in annotations:
     polygons = [r['shape_attributes'] for r in a['regions']]
     objects = [s['region_attributes'] for s in a['regions']]
     num_ids = []
   for n in objects:
     print(one)
     print(n)
   try:
    if n['object'] == 'Person':
     num_ids.append(1)
    elif n['object'] == 'Car':
     num_ids.append(2)
    elif n['object'] == 'Dog':
     num_ids.append(3)
    elif n['object'] == 'Chair':
     num_ids.append(4)
   except:
    pass

   image_path = os.path.join(dataset_dir, a['filename'])
   image = skimage.io.imread(image_path)
   (height, width) = image.shape[:2]
   self.add_image(  
                'object',
                image_id=a['filename'],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids,
                )
# also change the return value of def load_mask()
   num_ids = np.array(num_ids, dtype=np.int32)              
############################################################
#  Training
############################################################

if __name__ == '__main__':
  import argparse

    # Parse command line arguments
  parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
  parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
  parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
  parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
  parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
  parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
  parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
  args = parser.parse_args()

    # Validate arguments
  if args.command == "train":
    assert args.dataset, "Argument --dataset is required for training"
  elif args.command == "splash":
    assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
      config = CustomConfig()
    else:
      class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
       GPU_COUNT = 1
       IMAGES_PER_GPU = 1
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
    if args.weights.lower() == "coco":
      weights_path = COCO_WEIGHTS_PATH
        # Download weights file
      if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
      weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
      weights_path = model.get_imagenet_weights()
    else:
      weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
      model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
      model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
      train(model)
    elif args.command == "splash":
      detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
      print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))