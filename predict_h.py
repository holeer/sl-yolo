from PIL import Image
from yolo import YOLO
import os
import glob

HEATMAP_DEFAULT = 'img/1038.png'
MODEL_ROOT_PATH = r'logs'
MAX_EPOCH = 100
OUTPUT_DIR = 'model_data/heatmaps'

if __name__ == "__main__":
    WIDTH = len(str(MAX_EPOCH))
    if HEATMAP_DEFAULT is not None and len(HEATMAP_DEFAULT)>0:
        img = HEATMAP_DEFAULT
    else: img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        for epoch in range(1,MAX_EPOCH+1):
            prefix = 'ep' + str(epoch).zfill(WIDTH)
            model_path = glob.glob(f"{MODEL_ROOT_PATH}/{prefix}*")[0]
            yolo = YOLO(model_path=model_path)
            save_path = os.path.join(OUTPUT_DIR,prefix+'.png')
            yolo.detect_heatmap(image, save_path)