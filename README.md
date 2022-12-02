# MDA_noisy_label_learning
Official implementation of the AAAI2023 Workshop paper : Model and Data Agreement for Learning with Noisy Labels

## Abstract
Learning with noisy labels is a vital topic for practical deep
learning as models should be robust to noisy open-world
datasets in the wild. The state-of-the-art noisy label learn-
ing approach JoCoR fails when faced with a large ratio of
noisy labels. Moreover, selecting small-loss samples can also
cause error accumulation as once the noisy samples are mis-
takenly selected as small-loss samples, they are more likely
to be selected again. In this paper, we try to deal with error
accumulation in noisy label learning from both model and data
perspectives. We introduce mean point ensemble to utilize a
more robust loss function and more information from unse-
lected samples to reduce error accumulation from the model
perspective. Furthermore, as the flip images have the same
semantic meaning as the original images, we select small-loss
samples according to the loss values of flip images instead of
the original ones to reduce error accumulation from the data
perspective. Extensive experiments on CIFAR-10, CIFAR-
100, and large-scale Clothing1M show that our method out-
performs state-of-the-art noisy label learning methods with
different levels of label noise. Our method can also be seam-
lessly combined with other noisy label learning methods to
further improve their performance and generalize well to other
tasks.

## Train

Train with 80\% label noise on CIFAR-100 with 4 GPUs
```key
python train_cifar100.py
```

