# Conditional Image Synthesis With Auxiliary Classifier GANs

See README_og.md for complete details. Forked from `https://github.com/gitlimlab/ACGAN-PyTorch`

## Example Usage:

### For MNIST zeros and ones

#### Improved GAN

```
CUDA_VISIBLE_DEVICES=1 python main.py --outf=/scratch0/ilya/locDoc/ACGAN/experiments/julytest13 --train_batch_size=32 --cuda --dataset mnist_subset --num_classes 2 --imageSize=32 --data_root=/scratch0/ilya/locDoc/data/mnist --eval_period 50 --nc 1 --ndf 32 --ngf 32 --GAN_lrD 0.0001 --g_loss feature_matching
```

#### Mary GAN

```
CUDA_VISIBLE_DEVICES=1 python main.py --outf=/scratch0/ilya/locDoc/ACGAN/experiments/julytest13 --train_batch_size=32 --cuda --dataset mnist_subset --num_classes 2 --imageSize=32 --data_root=/scratch0/ilya/locDoc/data/mnist --eval_period 50 --nc 1 --ndf 32 --ngf 32 --GAN_lrD 0.0001
```

#### activation maximization gan

```
CUDA_VISIBLE_DEVICES=1 python main.py --outf=/scratch0/ilya/locDoc/ACGAN/experiments/julytest7 --train_batch_size=32 --cuda --dataset=mnist_subset --imageSize=32 --data_root=/scratch0/ilya/locDoc/data/mnist --eval_period 50 --nc 1 --num_classes 2 --ndf 32 --ngf 32 --GAN_lrD 0.0001 --g_loss activation_maximization
```

#### complement GAN

```
CUDA_VISIBLE_DEVICES=1 python main.py --outf=/scratch0/ilya/locDoc/ACGAN/experiments/julytest9 --train_batch_size=32 --cuda --dataset=mnist_subset --imageSize=32 --data_root=/scratch0/ilya/locDoc/data/mnist --eval_period 50 --nc 1 --num_classes 2 --ndf 32 --ngf 32 --GAN_lrD 0.0001 --g_loss crammer_singer_complement --g_loss_aux confuse --g_loss_aux_weight 0.33 --confuse_margin 1.0
```


### Full MNIST

```
CUDA_VISIBLE_DEVICES=1 python main.py --outf=/scratch0/ilya/locDoc/ACGAN/experiments/julytest10 --train_batch_size=32 --cuda --dataset=mnist --imageSize=32 --data_root=/scratch0/ilya/locDoc/data/mnist --eval_period 50 --nc 1 --num_classes 10 --ndf 32 --ngf 32 --GAN_lrD 0.0001 --g_loss crammer_singer_complement --g_loss_aux confuse --g_loss_aux_weight 0.33 --confuse_margin 1.0
```


