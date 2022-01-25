# ArcFace

Face Recogntion trainer using ArcFace method. This repo is based from third party implementation of ArcFace which is [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

# How to Use
## Data preparation

You need to convert your data into the required format

```python
    python im2rec.py --list --resize 112 --recursive ./my_data IMG_DIR
```

```python
    python3 img2rec.py ./my_lst lst_DIR ./my_data IMG_DIR
```

```python
    python dir2lst.py
```

## Training

Follow the .ipynb notebooks after you've done with data preparation