# LabelaHerdinMinutes

This repository contains the source code that accompanies our paper "Label a Herd in Minutes: Individual Holstein-Friesian Cattle Identification" at: https://arxiv.org/abs/2105.01938. (watiting archive)

## Depedencies
1) Clone this repository.
2) Install any missing requirements via pip or conda: [numpy](https://pypi.org/project/numpy/), [PyTorch](https://pytorch.org/), [OpenCV](https://pypi.org/project/opencv-python/), [Pillow](https://pypi.org/project/Pillow/), [tqdm](https://pypi.org/project/tqdm/), [sklearn](https://pypi.org/project/scikit-learn/), [seaborn](https://pypi.org/project/seaborn/). This repository requires python 3.6+
3) Use a computer with GPU for training.

## Usage
To replicate the results obtained in our paper, please use the [Cows2021](https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7) dataset. 
The training data in in Cows2021 Dataset/Sub-levels/Identification/Train [Download](https://data.bris.ac.uk/data/dataset/44ec2bfeda051bf39f8357d237db03af). The test data in Cows2021 Dataset/Sub-levels/Identification/Test [Download](https://data.bris.ac.uk/data/dataset/9ce27d05a89d12e4375986946fed59e5).




### Model Re-training
Replace the path of the training dataset in config.py with your own path. To train the model, use python `train.py -h` to get help with setting command line arguments. Examples of runing the code with fully supervised mode or metric learning mode are in `run.txt`. The training dataset is in `Cows2021 Sub-levels/Train`.
Merge tracklets from the training dataset: `Datasets/make_retrain` Replace the merge dataset to the origianl one and run `train.py` to finetune the model.


### Test
To test a trained model, see `run.txt`.
We excluded poor quality data from the test data of Cows2021. Please see `Datasets/data_list.txt` to find the test data using in our paper and run `Datasets/selection.py` to move these data to a new folder.


## Citation
Consider citing ours and William's works in your own research if this repository has been useful:

```
@article{gao2021towards,
  title={Towards Self-Supervision for Video Identification of Individual Holstein-Friesian Cattle: The Cows2021 Dataset},
  author={Gao, Jing and Burghardt, Tilo and Andrew, William and Dowsey, Andrew W and Campbell, Neill W},
  journal={arXiv preprint arXiv:2105.01938},
  year={2021}
}

@article{andrew2020visual,
  title={Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning},
  author={Andrew, William and Gao, Jing and Campbell, Neill and Dowsey, Andrew W and Burghardt, Tilo},
  journal={arXiv preprint arXiv:2006.09205},
  year={2020}
}
```
