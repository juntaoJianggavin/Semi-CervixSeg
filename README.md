# Semi-CervixSeg
Official PyTorch implementation of the paper "Semi-CervixSeg: A Multi-stage Training Strategy for Semi-Supervised Cervical Segmentation".

## Prepare data
The dataset used in this paper is: [Dataset for Fetal Ultrasound Grand Challenge: Semi-Supervised Cervical Segmentation (FUGC 2025)](https://zenodo.org/records/14305302).The dataset can be accessible after signing the data-sharing agreement and sending it to the organizer (fugc.isbi25@gmail.com).

The directory structure of the whole project is as follows:
```
── Semi-CervixSeg
│   ├──train_initial.py
│   ├──train_subsequent.py
│   ├──...
│   └──train
│        └── unlabeled_data
│        │        └── images
│        │             ├──0001.png
│        │             ├──0002.png
│        │             └──...
│        │
│        └──labeled_data
│                  ├── images
│                  │     ├──0001.png
│                  │     ├──0002.png
│                  │     └──...
│                  │
│                  └── labels
│                        ├──0001.png
│                        ├──0002.png
│                        └──...
│
```


## Usage
Initial Stage: Contrastive Learning for the Unlabeled data.
```bash
python train_initial.py
```

Subsequent Stages: Pseudo-label Generation and Refinement.
```bash
python train_subsequent.py
```
