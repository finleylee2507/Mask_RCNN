import os
import json
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='BDD100K to COCO format')
    parser.add_argument( 
          "-l", "--label_dir", 
          default=r"D:\Computer science\Mask_RCNN-master\samples\car\annotations", #where the downloaded JSON annotation file is 
          help="root directory of BDD label Json files",
    )
    parser.add_argument(
          "-s", "--save_path",
          default=r"D:\Computer science\Mask_RCNN-master\samples\car\bdd_to_coco_converted_labels", #where you want the converted file stored 
          help="path to save coco formatted label file",
    )
    return parser.parse_args()


def bdd2coco_detection(id_dict, labeled_images, fn):

    images = list()
    annotations = list()
    counter = 0
    for i in tqdm(labeled_images): #smart progress meter
        counter += 1
        image = dict()
        image['file_name'] = i['name']
        image['height'] = 720
        image['width'] = 1280

        image['id'] = counter

        empty_image = True
        for l in i['labels']:
            annotation = dict()
            if l['category'] in id_dict.keys():
                if(l['category'] == "drivable area"):
                    print(l['category'])
                    annotation['polygons'] = l['poly2d']
                else:
                    print(l['category'])
                    empty_image = False
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = image['id']
                    x1 = l['box2d']['x1']
                    y1 = l['box2d']['y1']
                    x2 = l['box2d']['x2']
                    y2 = l['box2d']['y2']
                    annotation['bbox'] = [x1, y1, x2-x1, y2-y1]
                    annotation['area'] = float((x2 - x1) * (y2 - y1))
                    annotation['category_id'] = id_dict[l['category']]
                    annotation['ignore'] = 0
                    annotation['id'] = l['id']
                    annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                    annotations.append(annotation)
        if empty_image:
            continue

        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('saving...')
    json_string = json.dumps(attr_dict)
    with open(fn, "w") as file:
        file.write(json_string)

#if file being run as the main program 
if __name__ == '__main__':

    args = parse_arguments()

    attr_dict = dict()
    attr_dict["categories"] = [
        {"supercategory": "none", "id": 1, "name": "person"},
        {"supercategory": "none", "id": 2, "name": "rider"},
        {"supercategory": "none", "id": 3, "name": "car"},
        {"supercategory": "none", "id": 4, "name": "bus"},
        {"supercategory": "none", "id": 5, "name": "truck"},
        {"supercategory": "none", "id": 6, "name": "bike"},
        {"supercategory": "none", "id": 7, "name": "motor"},
        {"supercategory": "none", "id": 8, "name": "traffic light"},
        {"supercategory": "none", "id": 9, "name": "traffic sign"},
        {"supercategory": "none", "id": 10, "name": "train"},
        {"supercategory": "none", "id": 11, "name": "drivable area"}
    ]

    attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
    print(attr_id_dict)
    #{'person': 1, 'rider': 2, 'car': 3, 'bus': 4, 'truck': 5, 'bike': 6, 'motor': 7, 'traffic light': 8, 'traffic sign': 9, 'train': 10, 'drivable area': 11}
    
    #create BDD training set detections in COCO format
    print('Loading training set...')
    with open(os.path.join(args.label_dir, 'custom_labels_images_train.json')) as f: #open downloaded JSON file 
        train_labels = json.load(f)
    print('Converting training set...')

    

    out_fn = os.path.join(args.save_path, 'custom_labels_images_det_coco_train.json')
    bdd2coco_detection(attr_id_dict, train_labels, out_fn)

    # print('Loading validation set...')
    # #create BDD validation set detections in COCO format
    # with open(os.path.join(args.label_dir,
    #                        'bdd100k_labels_images_val.json')) as f:
    #     val_labels = json.load(f)
    # print('Converting validation set...')

    # out_fn = os.path.join(args.save_path,
    #                       'bdd100k_labels_images_det_coco_val.json')
    # bdd2coco_detection(attr_id_dict, val_labels, out_fn)
