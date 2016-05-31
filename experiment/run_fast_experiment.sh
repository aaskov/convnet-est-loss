#!/bin/bash

echo 'Runnning experiment'

python experiment_estimate_loss.py \
	--proto ..model/cifar10/cifar10_full.prototxt \
	--model ..model/cifar10/bkp_cifar10_full_converge_002/cifar10_full_iter_100000.caffemodel.h5 \
	--meanfile ..model/cifar10/mean.binaryproto \
	--data ..data/cifar10/cifar10_train_lmdb/ \
	--layer conv1 \
	--iterations 500 \
	--damage 0.25

echo 'Done'
