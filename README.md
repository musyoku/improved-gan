# Improved Techniques for Training GANs

code for the [paper](https://arxiv.org/abs/1606.03498)

[実装について](http://musyoku.github.io/2016/12/23/Improved-Techniques-for-Training-GANs/)

### Requirements

- Chainer 1.18

# Animeface

[See result (7MB .jpg)](http://musyoku.github.io/images/post/2016-12-22/compare.jpg)

## Analogies

### DCGAN

![image](https://musyoku.github.io/images/post/2016-12-22/analogy.jpg)
[larger image](http://musyoku.github.io/images/post/2016-12-22/analogy.jpg)

### DCGAN + Feature matching

![image](https://musyoku.github.io/images/post/2016-12-22/analogy_fm.jpg)
[larger image](http://musyoku.github.io/images/post/2016-12-22/analogy_fm.jpg)

### DCGAN + Minibatch discrimination

![image](https://musyoku.github.io/images/post/2016-12-22/analogy_md.jpg)
[larger image](http://musyoku.github.io/images/post/2016-12-22/analogy_md.jpg)

### DCGAN + Minibatch discrimination + Feature matching

![image](https://musyoku.github.io/images/post/2016-12-22/analogy_fm_md.jpg)
[larger image](http://musyoku.github.io/images/post/2016-12-22/analogy_fm_md.jpg)

# Semi-supervised MNIST Classification

| data | # |
|:--:|:--:|
| labeled | 100 |
| unlabeled | 49900 |
| validation | 10000 |

![accuracy](https://musyoku.github.io/images/post/2016-12-22/accuracy.png?raw=true)
