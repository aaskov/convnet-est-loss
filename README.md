ConvNet estimation of loss
=============
The code used in the 'Estimating the expected loss value in a Deep Neural Network using Gauss-Newton hessian approximation' paper.

Requirements
-----
* Caffe (http://caffe.berkeleyvision.org/) -- Latest version is recommended
* NumPy (http://www.numpy.org/) -- `sudo apt-get install python-numpy`
* Matplotlib (http://matplotlib.org/) -- `sudo apt-get install python-matplotlib`

Installation
-----
```bash
cd ~ && git clone https://github.com/aaskov/convnet-est-loss.git
```

Usage
-----
You will need a local copy of the CIFAR-10 data set in the `data/cifar10/` folder as well as a trained caffe model in the `model/cifar10/` folder. The trained caffe model requires a `*.prototxt`, `*.caffemodel` and a `*.mean.binaryproto` file.
```bash
cd ~/convnet-est-loss/experiment
python experiment_estimate_loss.py --proto ..model/cifar10/cifar10_full.prototxt --model ..model/cifar10/cifar10_full_iter_100000.caffemodel.h5 --meanfile ..model/cifar10/mean.binaryproto --data ..data/cifar10/cifar10_train_lmdb/ --layer conv1
```
For more detail see
```bash
python experiment/experiment_damage.py --help
python experiment/experiment_estimate_loss.py --help
```

License
-----
MIT
