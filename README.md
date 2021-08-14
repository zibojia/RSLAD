# RSLAD
This is the official code for "Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better". The paper can be download [here](https://arxiv.org).

### Prerequisites
* Python 3.7
* Pytorch 1.3
* CUDA 10.1
* numpy 1.19

### Code Implementation
**Baseline Implementation**

* For the training for [SAT](https://arxiv.org/pdf/1706.06083.pdf)(which is also known as Madry), [TRADES](https://arxiv.or/pdf/1901.08573.pdf) and [ARD](https://arxiv.org/pdf/1905.09747.pdf), we use the code released by official. The codes are available here([SAT](https://github.com/MadryLab/cifar10_challenge),[TRADES](https://github.com/yaodongyu/TRADES) and [SAT](https://github.com/goldblum/AdversariallyRobustDistillation)).

**Teacher Implementation**

* For the teacher model, the WideResNet-34-10 TRADES-pretrained is downloaded [here](https://drive.google.com/file/d/10sHvaXhTNZGz618QmD5gSOAjO3rMzV33/view) and WideResNet-70-16 is downloaded [here](https://drive.google.com/file/d/10sHvaXhTNZGz618QmD5gSOAjO3rMzV33/view).

**Student Implementation**

* For the student model, we use the loss defined in RSLAD to compare with ARD and IAD. Note the KL implementation issue in RSLAD, ARD and IAD. The orignal KL divergence is defined as: <img src="http://chart.googleapis.com/chart?cht=tx&chl= D_{KL} = -\sum_{i=0}^{n} P(i)ln(\frac{Q(i)}{P(i)})" style="border:none;"> and  the pytorch default implementation is defined as : <img src="http://chart.googleapis.com/chart?cht=tx&chl= D_{KL} = -\frac{1}{n}\cdot\sum_{i=0}^{n} P(i)ln(\frac{Q(i)}{P(i)})" style="border:none;">. For CIFAR-10, **n=10**, for CIFAR-100, **n=100**, which means the KL used for CIFAR-100 is **10 times smaller** than CIFAR-10 in pytorch default implementation. Thus, we multiply 10 for KL used for CIFAR-100 to keep consistent with CIFAR-10. 

### Model Pretrained Download
* We provided the pretrained models. It can be download [here](https://www.google.com/drive/)

### Citation
```
@inproceedings{zi2021revisiting, 
	author = {Bojia Zi and Shihao Zhao and Xingjun Ma and Yu-Gang Jiang}, 
	title = {Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better}, 
	booktitle = {International Conference on Computer Vision},
	year = {2021}
}
```
