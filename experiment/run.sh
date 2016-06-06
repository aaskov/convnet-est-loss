#!/bin/bash

echo 'Runnning the experiment...'

python experiment_estimate_loss.py \
	--proto /home/aaskov/caffe/examples/cifar10/cifar10_full.prototxt \
	--model ../model/cifar10_full_iter_140000.caffemodel \
	--meanfile /home/aaskov/caffe/examples/cifar10/mean.binaryproto \
	--data /home/aaskov/caffe/examples/cifar10/cifar10_train_lmdb/ \
	--layer conv1 \
	--iterations 50000 \
	--damage 0.25 \
	--repeat 20

echo 'Done.'
