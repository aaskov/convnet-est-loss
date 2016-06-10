#!/bin/bash

echo 'Runnning the experiment...'

python experiment_estimate_loss.py \
	--proto ../model/cifar10/cifar10_full.prototxt \
	--model ../model/cifar10/cifar10_full_iter_200000.caffemodel \
	--meanfile ../model/cifar10/mean.binaryproto \
	--data ..data/cifar10/cifar10_train_lmdb/ \
	--layer conv1 \
	--iterations 250 \
	--damage 0.25 \
	--repeat 1

echo 'Done.'
