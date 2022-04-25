# LabelaHerdinMinutes

This repository contains the source code that accompanies our paper "Label a Herd in Minutes: Individual Holstein-Friesian Cattle Identification" at: https://arxiv.org/abs/2105.01938(watiting).

## Installation
1) Clone this repository.
2) Install any missing requirements via pip or conda: [numpy](https://pypi.org/project/numpy/), [PyTorch](https://pytorch.org/), [OpenCV](https://pypi.org/project/opencv-python/), [Pillow](https://pypi.org/project/Pillow/), [tqdm](https://pypi.org/project/tqdm/), [sklearn](https://pypi.org/project/scikit-learn/), [seaborn](https://pypi.org/project/seaborn/). This repository requires python 3.6+
3) Instead of installing `pycocotools`, use the pycocotools in this repository.

## Usage
To replicate the results obtained in our paper, please download the Cows2021 dataset at: [download](https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7).
To train the model, use python train.py -h to get help with setting command line arguments. A minimal example would be python train.py --out_path=output/ --folds_file=datasets/OpenSetCows2020/splits/10-90.json.
format:...

To test a trained model, use python test.py -h. A minimal example would be python test.py --model_path=output/fold_0/best_model_state.pkl --folds_file=datasets/OpenSetCows2020/splits/10-90.json --save_path=output/fold_0/.


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
