# Semi-CervixSeg
Official PyTorch implementation of the paper "Semi-CervixSeg: A Multi-stage Training Strategy for Semi-Supervised Cervical Segmentation".

## Usage
### InitialStage: Contrastive Learning for the Unlabeled data
```bash
python train_initial.py
```

### Subsequent Stages: Pseudo-label Generation and Refinement
```bash
python train_subsequent.py
```
