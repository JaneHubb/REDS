# REDS
Retrieval-Augmented and Debiased Sequential Recommendation with Heterogeneous Item Embeddings (REDS)
 
### Environments
* Python 3.7+
* PyTorch 1.12+
* CUDA 11.6+

### Setup
Install the required packages:
```pip install -r requirements.txt```

### Run the code
```python run.py```

When you first run, the script will automatically download the dataset specified in ```config.yaml```.
After the download is complete, run the same command again to start training and inference.

## Acknowledgment
This project is based on [SASRec](https://github.com/kang205/SASRec), and [RecBole](https://github.com/RUCAIBox/RecBole). We are grateful for their outstanding contributions.
