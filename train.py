import warnings
from argparse import ArgumentParser
from ultralytics import RTDETR

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./ultralytics/cfg/models/rt-detr/Daw-DETR.yaml')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='exp')
    args = parser.parse_args()
    model = RTDETR(args.config)
    model.train(
        data=args.data,
        cache=False,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
    )

