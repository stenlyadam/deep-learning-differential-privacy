# Deep Learning with Differential Privacy (DP-SGD)

This project implements Differential Privacy in Deep Learning using DP-SGD based on the paper: Abadi et al., Deep Learning with Differential Privacy (2016). The experiment evaluates how privacy protection affects model performance using the MNIST dataset.

## Course Name: Data Privacy and Security (1142CS5164701)
### Group Name: SecureBytes
Member:
1. Alim Misbullah D11415803	
2. Laina Farsiah D11415802
3. Stenly Ibrahim Adam D11215809
4. Aurelio Naufal Effendy M11415802

## Objectives
* Implement DP-SGD using TensorFlow Privacy
* Compare baseline vs privacy-preserving models
* Analyze privacy-utility tradeoff
* Evaluate privacy using epsilon (ε)

## Method
### Baseline Model
* Standard CNN trained with SGD
### DP Model
* Gradient clipping
* Gaussian noise addition
* Privacy accounting using ε

## Results Summary
| Model | Noise Multiplier | Test Accuracy | Epsilon | Training Time (s) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.0 | 98.31% | N/A | 9.02 |
| DP-SGD | 0.5 | 88.36% | 81.116 | 156.92 |
| DP-SGD | 1.0 | 91.77% | 30.127 | 187.95 |
| DP-SGD | 1.5 | 87.15% | 17.665 | 192.94 |
| DP-SGD | 2.0 | 91.06% | 12.302 | 197.81 |
| DP-SGD | 2.5 | 89.23% | 9.368 | 196.37 |
| DP-SGD | 3.0 | 90.12% | 7.532 | 196.8 |

### Key Insight
* Strong privacy can be achieved with minimal accuracy loss
* DP introduces a tradeoff between privacy and performance

## Visualizations
### Accuracy Comparison
Comparison of test accuracy between baseline and DP models.

<p align="center">
  <img src="results/accuracy_comparison.png" width="500">
</p>

### Privacy-Utility Tradeoff
Relationship between epsilon (privacy level) and model accuracy.

<p align="center">
  <img src="results/privacy_utility_tradeoff.png" width="500">
</p>

## Setup
### Requirements
```
conda create -n dataprivacy_tf
source activate dataprivacy_tf
pip install -r requirements.txt
```

### Run Experiment
```
python code/hw2_dp_mnist.py
```

### Project Structure
```
code/        → training scripts
results/     → experiment outputs
report/      → final report
```
🔐 Privacy Explanation

DP-SGD protects training data by:

clipping gradients to limit individual influence
adding Gaussian noise to hide data contribution

Privacy is measured using epsilon (ε):

smaller ε → stronger privacy
💻 Environment
TensorFlow 2.15.1
TensorFlow Privacy 0.9.0
CUDA 12.2
GPU: NVIDIA RTX 2080 Ti
👥 Team
Alim Misbullah
Laina Farsiah
Stenly Ibrahim Adam
Aurelio Naufal Effendy
📚 References
Abadi et al. (2016)
TensorFlow Privacy Documentation
MNIST Dataset
