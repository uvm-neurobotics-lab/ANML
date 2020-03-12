# ANML: Learning to Continually Learn (ECAI 2020)

[arXiv Link](https://arxiv.org/abs/2002.09571)

Continual lifelong learning requires an agent or model to learn many sequentially ordered tasks, building on previous knowledge without catastrophically forgetting it. Much work has gone towards preventing the default tendency of machine learning models to catastrophically forget, yet virtually all such work involves manually-designed solutions to the problem. We instead advocate meta-learning a solution to catastrophic forgetting, allowing AI to learn to continually learn. Inspired by neuromodulatory processes in the brain, we propose A Neuromodulated Meta-Learning Algorithm (ANML). It differentiates through a sequential learning process to meta-learn an activation-gating function that enables context-dependent selective activation within a deep neural network. Specifically, a neuromodulatory (NM) neural network gates the forward pass of another (otherwise normal) neural network called the prediction learning network (PLN). The NM network also thus indirectly controls selective plasticity (i.e. the backward pass of) the PLN. ANML enables continual learning without catastrophic forgetting at scale: it produces state-of-the-art continual learning performance, sequentially learning as many as 600 classes (over 9,000 SGD updates). 

## How to Run 

First, install [Anaconda](https://docs.continuum.io/anaconda/install/linux/) for Python 3 on your machine.

Next, install PyTorch and Tensorboard

```
pip install torch
pip install tensorboardX
```

Then clone the repository:

```
git clone https://github.com/shawnbeaulieu/ANML.git
```

Meta-train your network(s). To modify the network architecture, see modelfactory.py in the model folder. Depending on the architecture you choose, you may have to change how the data is loaded and/or preprocessed. See omniglot.py and task_sampler.py in the datasets folder.

```
python mrcl_classification.py --rln 7 --meta_lr 0.001 --update_lr 0.1 --name mrcl_omniglot --steps 20000 --seed 9 --model_name "Neuromodulation_Model.net"
```

Evaluate your trained model:

```
python evaluate_classification.py --rln 13  --model Neuromodulation_Model.net --name Omni_test_traj --runs 10

```

### Prerequisites

Python 3
PyTorch 1.4.0
Tensorboard

## Built From

* [OML/MRCL](https://github.com/khurramjaved96/mrcl)

