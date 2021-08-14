<!doctype html>
<html>
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" media="all" href="normalize.css">
    <link rel="stylesheet" media="all" href="core.css">
    <link rel="stylesheet" media="all" href="style.css">
    <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
</head>
<body data-document>&nbsp;</body>
</html>


# RSLAD
This is the official code for "Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better". The paper can be download [here](https://arxiv.org).

**Paper Implementation**

For the training for [SAT](https://arxiv.org/pdf/1706.06083.pdf)(which is also known as Madry), [TRADES](https://arxiv.or/pdf/1901.08573.pdf) and [ARD](https://arxiv.org/pdf/1905.09747.pdf), we use the code released by official. The codes are available here([SAT](https://github.com/MadryLab/cifar10_challenge),[TRADES](https://github.com/yaodongyu/TRADES) and [SAT](https://github.com/goldblum/AdversariallyRobustDistillation)).

Note the KL implementation issue. Since the KL is defined as: <img src="http://chart.googleapis.com/chart?cht=tx&chl= D_{KL} = -\sum_{i=0}^{n} P(i)ln(\frac{Q(i)}{P(i)})" style="border:none;">, the pytorch default implementation: <img src="http://chart.googleapis.com/chart?cht=tx&chl= D_{KL} = -\frac{1}{n}\cdot\sum_{i=0}^{n} P(i)ln(\frac{Q(i)}{P(i)})" style="border:none;">



