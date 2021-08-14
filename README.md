# RSLAD
This is the official code for "Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better". The paper can be download [here](https://arxiv.org).

### Code Implementation

For the training for [SAT](https://arxiv.org/pdf/1706.06083.pdf)(which is also known as Madry), [TRADES](https://arxiv.or/pdf/1901.08573.pdf) and [ARD](https://arxiv.org/pdf/1905.09747.pdf), we use the code released by official. The codes are available here([SAT](https://github.com/MadryLab/cifar10_challenge),[TRADES](https://github.com/yaodongyu/TRADES) and [SAT](https://github.com/goldblum/AdversariallyRobustDistillation)).

Note the KL implementation issue. The KL divergence is defined as: <img src="http://chart.googleapis.com/chart?cht=tx&chl= D_{KL} = -\sum_{i=0}^{n} P(i)ln(\frac{Q(i)}{P(i)})" style="border:none;">, the pytorch default implementation: <img src="http://chart.googleapis.com/chart?cht=tx&chl= D_{KL} = -\frac{1}{n}\cdot\sum_{i=0}^{n} P(i)ln(\frac{Q(i)}{P(i)})" style="border:none;">. For CIFAR-10, **n=10**, for CIFAR-100, **n=100**, which means the the KL used for CIFAR-100 is **10 times smaller** than CIFAR-10 in pytorch default implementation. Thus, we multiply 10 for CIFAR-100 to keep consistent with CIFAR-10. 



