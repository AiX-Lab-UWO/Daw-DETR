# Daw-DETR: Vision-Language-Enhanced Real-Time Object Detection in Adverse Weather for Autonomous Driving
## Introduction
This repository hosts implementations for the "Daw-DETR: Vision-Language-Enhanced Real-Time Object Detection in Adverse Weather for Autonomous Driving" submitted paper.

## Usage

### Installation
```bash
git clone https://github.com/AiX-Lab-UWO/Daw-DETR.git
conda env create -f environment.yaml
conda activate DawDETR
```

### Data Preparation
* Step 1: Download the dataset from the below link:
[BDD100K](https://drive.google.com/file/d/1R1jBgYlU2UQoF757XWGqLYLAHPqxh30g/view?usp=sharing)

* Step 2: Organize the downloaded files in the following way.
```bash
├─ Dataset
│   └─ images
│       ├─ test
│       ├─ train
│       └─ val
│   └─ labels
│       ├─ test
│       ├─ train
│       └─ val
│   └─ data.yaml
├─ Daw-DETR
```
- `images`: Contains subfolders for `train`, `val`, and `test` images.
- `labels`: Contains corresponding annotation files.
- `data.yaml`: A configuration file specifying dataset paths and class details.

**Note:** the original BDD100K annotation format differs from Ultralytics YOLO format. We have standardized all annotations to ensure compatibility.

### Train
The training results will be stored at `runs/train`.
```bash
python train.py --data ./Dataset/data.yaml --imgsz 640 --epochs 150 --patience 8 --batch 8 --device 0 --project runs/train --name exp
```

### Evaluation
To evaluate the trained model, use the following command:
```bash
python evaluate.py --checkpoint /path/to/your/checkpoint --data ./Dataset/data.yaml --imgsz 640 --batch 8
```

## Acknowledgement
This repository is built using the [ultralytics](https://github.com/ultralytics/ultralytics) repository.

## License
This project is licensed under the AGPL-3.0 License. See the [LICENSE](LICENSE) file for more details.
