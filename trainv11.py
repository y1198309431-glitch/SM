import os

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

data_yaml_path = r'./datasets/data.yaml'

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')

    results = model.train(
        data=data_yaml_path,
        imgsz=640,
        epochs=700,
        batch=20,
        lr0=0.01,
        workers=0,
        project='runs/V11train',
        name='exp_official',
        mosaic=1.0,
        scale=0.5,
        mixup=0.1,
        copy_paste=0.1,
        device='0,1'
    )
