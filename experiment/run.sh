#!/bin/bash

echo 'Runnning the experiment...'

# Layer 'conv1'
python experiment_damage.py \
	--proto ../model/cifar10/cifar10_full.prototxt \
	--model ../model/cifar10/cifar10_full_iter_200000.caffemodel \
	--meanfile ../model/cifar10/mean.binaryproto \
	--data ..data/cifar10/cifar10_train_lmdb/ \
	--layer conv1 \
	--iterations 50000 \
	--damage 1.0 \
	--repeat 0

# Layer 'conv2'
python experiment_damage.py \
	--proto ../model/cifar10/cifar10_full.prototxt \
	--model ../model/cifar10/cifar10_full_iter_200000.caffemodel \
	--meanfile ../model/cifar10/mean.binaryproto \
	--data ..data/cifar10/cifar10_train_lmdb/ \
	--layer conv1 \
	--iterations 50000 \
	--damage 1.0 \
	--repeat 0


# Layer 'conv3'
python experiment_damage.py \
	--proto ../model/cifar10/cifar10_full.prototxt \
	--model ../model/cifar10/cifar10_full_iter_200000.caffemodel \
	--meanfile ../model/cifar10/mean.binaryproto \
	--data ..data/cifar10/cifar10_train_lmdb/ \
	--layer conv1 \
	--iterations 50000 \
	--damage 1.0 \
	--repeat 0


# Layer 'ip1'
python experiment_damage.py \
	--proto ../model/cifar10/cifar10_full.prototxt \
	--model ../model/cifar10/cifar10_full_iter_200000.caffemodel \
	--meanfile ../model/cifar10/mean.binaryproto \
	--data ..data/cifar10/cifar10_train_lmdb/ \
	--layer conv1 \
	--iterations 50000 \
	--damage 1.0 \
	--repeat 0


echo 'Done.'
