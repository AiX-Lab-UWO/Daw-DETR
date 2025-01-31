import warnings
import torch
from ultralytics import RTDETR
from argparse import ArgumentParser

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--name', type=str, default='exp')
    args = parser.parse_args()
    model_name = args.checkpoint.split('/')[-1].split('.')[0]

    print(f"Evaluating model: {model_name}")
    model = RTDETR(args.checkpoint)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    precision = metrics.box.mp
    recall = metrics.box.mr
    mAP_05 = metrics.box.map50
    mAP_05_095 = metrics.box.map
    total_speed = sum(metrics.speed.values()) / 1000
    fps = 1 / total_speed
    print(f"Results for {model_name}:")
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("mAP@0.5: {:.2f}%".format(mAP_05 * 100))
    print("mAP@0.5:0.95: {:.2f}%".format(mAP_05_095 * 100))
    print("Speed (FPS): {:.2f}".format(fps))
