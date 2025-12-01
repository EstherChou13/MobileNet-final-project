# MobileNet-final-project
MobileNet Reproduction Project

ECBM E4040 Deep Learning and Neural Networks — Fall 2025

This repository contains an initial implementation for reproducing the core ideas of
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
(Howard et al., arXiv:1704.04861).

This first commit includes:
A minimal MobileNet (v1) implementation in TensorFlow/Keras
Training script using CIFAR-10
Adjustable width multiplier α
Placeholder directories for experiments and logs
Project structure ready for further extension
We will later:
Add full experiments for α = {1.0, 0.75, 0.5, 0.25}
Add experiments for different input resolutions
Compare model size, FLOPs, accuracy
Add plots and detailed documentation
Write final project report

~ Quick Start
pip install -r requirements.txt
python src/train.py --alpha 1.0

~ Repository Structure
mobilenet-reproduction/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── mobile_net.py      # minimal MobileNet implementation
│   ├── train.py           # training loop on CIFAR-10
│   ├── utils.py
│
├── experiments/
│   ├── logs/              # training log files
│   ├── results_placeholder.txt
│   └── plots/
│
└── data/                  # CIFAR-10 auto-downloads here

~ Current Status
More experiments and analysis will be added over the next two weeks.
