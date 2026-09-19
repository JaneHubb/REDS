# REDS: Learning Popularity-Aware User Representations for Sequential Recommendation via Retrieval-Augmentation

[![ACM TOIS](https://img.shields.io/badge/ACM%20TOIS-Accepted-blue.svg)](https://dl.acm.org/journal/tois)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![CUDA 11.6](https://img.shields.io/badge/CUDA-11.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)

This is the official implementation of the ACM Transactions on Information Systems (TOIS) paper **"Learning Popularity-Aware User Representations for Sequential Recommendation via Retrieval-Augmentation."**

## 🔍 Overview

**REDS** is a sequential recommendation framework designed to improve user representations by jointly modeling popularity-aware sequential preferences and collaborative information retrieved from related users. It incorporates frequency information into sequential representations and augments the target user representation with relevant collaborative signals through retrieval-based user modeling.

<!-- Add the overview figure here.
Example:
<div align="center">
  <img src="assets/overview.png" alt="Overview of REDS" width="900">
</div>
-->

## 🛠️ Environment Setup

### Requirements

- **Python**: 3.8+
- **CUDA**: 11.6+

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/JaneHubb/REDS.git
cd REDS/REDS
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 🚀 Training and Inference

Run REDS with:

```bash
python run.py
```

The dataset and experimental settings can be configured in `config.yaml`.

When `run.py` is executed for the first time, the dataset specified in `config.yaml` is automatically downloaded. After the download is complete, run the same command again to start training and inference.

## ⚙️ Configuration

Main experimental settings are specified in:

```text
config.yaml
```

Modify this file to select the dataset and adjust the corresponding training settings.

## 🙏 Acknowledgments

This project builds upon the following open-source projects:

- [**SASRec**](https://github.com/kang205/SASRec)
- [**RecBole**](https://github.com/RUCAIBox/RecBole)

We sincerely thank the authors for making their implementations publicly available.

## 📜 Citation

If you find this work helpful, please consider citing our paper.

> The complete BibTeX entry will be updated once the final ACM bibliographic record is available.
