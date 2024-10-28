import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from utils.common import split_even_odd_indices
from yolo import YOLO

MAP_MODE = 3
CLASSES_PATH = 'VOCdevkit/classes.txt' # single_class
# CLASSES_PATH = 'model_data/classes.txt' # multi_class
MAP_VIS = False # something wrong in this feature
VOCD_PATH = 'VOCdevkit'
IMAGE_ID_FILE = "ImageSets/Main/val.txt"
MAP_OUT_PATH = 'map_out'
MIN_OVERLAP      = 0.5
SCORE_THREHOLD   = 0.5

def __mode01(img_list,class_names):
    print("Load model.")
    yolo = YOLO(confidence=0.001, nms_iou=0.5)
    print("Load model done.")
    print("Get predict result.")
    if map_vis:
        MAP_VIS_DIR = os.path.join(MAP_OUT_PATH, "images-optional")
        if not os.path.exists(MAP_VIS_DIR):
            os.makedirs(MAP_VIS_DIR)
    image_id = 0
    for img in tqdm(img_list):
        image = Image.open(img)
        if map_vis:
            basename = os.path.basename(img)
            image.save(os.path.join(MAP_VIS_DIR, basename))
        yolo.get_map_txt(image_id, image, class_names, MAP_OUT_PATH)
        image_id += 1
    print("Prepare predict result done.")


def __mode02(coordinates_list,class_names):
    # xml to txt
    print("Prepare ground truth result.")
    for loop_id, coordinates in tqdm(enumerate(coordinates_list)):
        new_f_path = os.path.join(map_out_path, f"ground-truth/{loop_id}.txt")
        with open(new_f_path, "w") as new_f:
            xml_file = os.path.join(VOCdevkit_path, f"Annotations/{loop_id}.xml")
            root = ET.parse(xml_file).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult') != None:
                    difficult = obj.find('difficult').text
                    if int(difficult) == 1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox = obj.find('bndbox')
                left = bndbox.find('xmin').text
                top = bndbox.find('ymin').text
                right = bndbox.find('xmax').text
                bottom = bndbox.find('ymax').text
                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Prepare ground truth result done.")


if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，在门限值不同时，网络的Recall和Precision值是不同的。
    map计算结果中的Recall和Precision代表的是当预测时，门限置信度为0.5时，所对应的Recall和Precision值。

    此处获得的./map_out/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
    目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = MAP_MODE
    #-------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #-------------------------------------------------------#
    classes_path    = CLASSES_PATH
    #-------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #-------------------------------------------------------#
    min_overlap      = MIN_OVERLAP
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #-------------------------------------------------------#
    map_vis         = MAP_VIS
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = VOCD_PATH
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #-------------------------------------------------------#
    map_out_path = MAP_OUT_PATH

    txt_file = os.path.join(VOCD_PATH, IMAGE_ID_FILE)
    txt_segments = open(txt_file).read().strip().split()
    img_list, coordinates_list = split_even_odd_indices(txt_segments)

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        __mode01(img_list,class_names)

    if map_mode == 0 or map_mode == 2:
        __mode02(coordinates_list,class_names)

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(min_overlap, True, map_out_path, SCORE_THREHOLD)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
