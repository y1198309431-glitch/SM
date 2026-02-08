import warnings
warnings.filterwarnings('ignore')
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets
def parse_opt():
    parser = argparse.ArgumentParser()
    # 1.大家经常会遇到的报错
    # parser.add_argument('--anno_json', type=str, default=r'E:\yolo\yolov8v10\COCO\data.json',help='label coco json path')
    # parser.add_argument('--pred_json', type=str, default=r'E:\yolo\yolov8v10\runs\val\exp\predictions.json',help='pred coco json path')

    # 2.完整视频更新在YOLO涨点改进交流群里面，教大家如何解决这个报错
    parser.add_argument('--anno_json', type=str, default=r'E:\yolo\yolov11v12\COCO\data_val1.json', help='label coco json path')
    parser.add_argument('--pred_json', type=str, default=r'E:\yolo\yolov11v12\output2.json', help='pred coco json path')
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    tide = TIDE()
    tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    tide.summarize()
