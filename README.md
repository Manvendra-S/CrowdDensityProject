# Crowd Density Estimation

## Introduction
Crowd density estimation at cross sections is essential for understanding pedestrian movement patterns, improving traffic management, and ensuring public safety. This project presents a deep learning-based method using Multi-column Convolutional Neural Networks (M-CNN) to automatically calculate crowd density from images. The system combines computer vision techniques with deep learning algorithms to accurately assess crowd density. The ShanghaiTech dataset is utilized for training and testing the model, which demonstrates high accuracy and robustness.

The paper related to this project is published and can be accessed via the following DOI: [https://doi.org/10.1109/ICAC3N60023.2023.10541536](https://doi.org/10.1109/ICAC3N60023.2023.10541536).

## Getting Started

### Prerequisites
- Python 3.7 or higher

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/crowd-density-estimation.git
    cd crowd-density-estimation
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Dataset
The ShanghaiTech dataset is used for training and testing the M-CNN model. You can download the dataset from the following link:

[Download ShanghaiTech Dataset](https://www.kaggle.com/datasets/tthien/shanghaitech-with-people-density-map)

After downloading, extract the dataset into the `data/` directory of the project.

### Training the Model
To train the model, run:
```bash
python train.py
