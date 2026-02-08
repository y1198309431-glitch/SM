from ultralytics import YOLO
import warnings
import torch.nn as nn
warnings.filterwarnings('ignore')
# 模型配置文件
model_yaml_path = r'/tmp/pycharm_project_490/ultralytics/cfg/models/11/ultra-yolo11s.yaml'


#数据集配置文件
data_yaml_path = r'./datasets/data.yaml'
if __name__ == '__main__':
    model = YOLO(model_yaml_path)
    #训练模型
    results = model.train(data=data_yaml_path,
                          imgsz=640,
                          epochs=700,
                          batch=32,
                          lr0=0.01,
                          workers=2,
                          project='runs/V11train',
                          name='exp',
                          mosaic=1.0,
                          scale=0.5,
                          mixup=0.1,
                          copy_paste=0.1,
                          )




