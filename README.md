# Graph Neural Network for Vision (GNN and Analyze)

This project contains a graph neural network (GNN) model for vision tasks, along with
supporting training scripts and experiments. The original notebook-based workflow has
been converted into a Python entrypoint for easier reproducibility.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
bash run.sh --dataset-path /path/to/CODEBRIM
```

The CODEBRIM dataset is expected to follow this structure:

```
CODEBRIM/
  train/
  val/
  test/
  metadata/
    defects.xml
    background.xml
```

## Other scripts

- `meta_learning.py` — meta-learning experiments (CIFAR100/SVHN)
- `transferLearning.py` — transfer learning example for image classification

These scripts may require additional dependencies beyond `requirements.txt` based on
their specific experiment settings.
