# -*- coding: utf-8 -*-
"""
convnet-est-loss
"""
import numpy as np
import os
import sys
import lmdb
import time

# Setup Caffe
caffe_dir = '/home/aaskov/caffe/'
sys.path.insert(0, caffe_dir + 'python')
import caffe
caffe.set_mode_cpu()


class Network(object):
    """Network class module.

    This class contains all the necessary settings and functions to setup and
    run a convnet for experiments.

    Note:
        Text `self` text ``Args`` section.

    Args:
        prototxt (str): Path to the `.prototxt` file.
        caffemodel (str): Path to the `.caffemodel` file (trained network).
        meanfile (str): Path to the mean file `mean.binaryproto`.
        path_to_data (str): Path to the lmdb data file.
    """
    def __init__(self, prototxt, caffemodel, meanfile, path_to_data):
        self.prototxt = prototxt
        self.caffemodel = caffemodel
        self.meanfile = meanfile
        self.path_to_data = path_to_data

        # Setup and configure
        self.__mean_image__()
        self.__config__()

    def forward(self, maxrun=10000, k=5, post=1000):
        """Network forward push.

        Push data set through network.

        Note:
          Use `maxrun` to determine how many images to push through.

        Args:
          maxrun (Optinal[int]): Maximum nnumber of images to pass through.
          k (Optional[int]): Define how many predictions to include in `top k`.
          post (Optinal[int]): When to post.
        """
        lmdb_env = lmdb.open(self.path_to_data)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe.proto.caffe_pb2.Datum()

        # Loop stats
        count = 0
        accuracy = 0.0
        top_1_error = 0.0
        top_k_error = 0.0
        loss = 0.0

        for key, value in lmdb_cursor:
            t = time.time()

            # Extract the image from the LMDB dataset
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            label = int(datum.label)
            input_image = caffe.io.datum_to_array(datum)
            input_image = input_image.astype(np.uint8)

            # Insert into the network
            self.caffe_net.blobs['data'].data[...] = input_image
            net_out = self.caffe_net.forward()

            # Evaluate
            predict_label = int(net_out['prob'][0].argmax(axis=0))
            output_prob = self.caffe_net.blobs['prob'].data[0]
            top_inds = output_prob.argsort()[::-1][:k]

            # Get loss
            target = np.zeros_like(net_out['prob'][0])
            target[label] = 1.0  # Suggested by Caffe community
            loss += target.dot(np.log(net_out['prob'][0]))

            # Stats
            iscorrect = predict_label == label
            accuracy += (1 if iscorrect else 0)
            top_1_error += (0 if label == top_inds[0] else 1)
            top_k_error += (0 if label in top_inds[:k] else 1)

            count += 1
            if count % post == 0:
                text = 'Running %i. Acc: %0.5f. Time: %0.5f'
                print text % (count, accuracy/count, post*(time.time()-t))
            if count > maxrun:
                break

        # Normalize
        accuracy /= count
        top_1_error /= count
        top_k_error /= count
        loss /= count

        return((accuracy, top_1_error, top_k_error, (-1)*loss))

    def get_hessian(self, wanted_layers, maxrun=10000, post=1000, with_bias=False):
        """Approx. hessian estimate.

        Text text text text text text text text text text text text text text
        text text text text.

        Args:
          wanted_layers (List[str]): List containing layer names (eg [`conv1`]).
          maxrun (Optional[int]): Number of images to use.
          post (Optional[int]): When to post.
          with_bias (Optinal[bool]): Toggle use of bias parameters.
        """
        lmdb_env = lmdb.open(self.path_to_data)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe.proto.caffe_pb2.Datum()

        # For pre-config
        diff = self.__gradients__(wanted_layers, with_bias)
        trace = np.zeros_like(diff)

        # Loop over images
        count = 0
        for key, value in lmdb_cursor:
            t = time.time()

            # Extract the image from the LMDB dataset
            datum.ParseFromString(value)
            label = int(datum.label)
            input_image = caffe.io.datum_to_array(datum)

            # Insert into the network and pass thorugh
            self.caffe_net.blobs['data'].data[...] = input_image
            self._forward = self.caffe_net.forward()

            # Backward to compute gradients
            # Suggested from the Caffe-users google forum
            self.caffe_net.blobs['prob'].diff[0][label] = 1
            self._backward = self.caffe_net.backward(**{self.caffe_net.outputs[0]: self.caffe_net.blobs['prob'].diff})

            # Calculate the derivatices and store the approx Hessian
            diff = self.__gradients__(wanted_layers, with_bias)
            trace += np.power(diff, 2)

            count += 1
            if count > maxrun:
                break
            if count % post == 0:
                print 'Running %i. Time: %0.5f' % (count, post*(time.time()-t))
                sys.stdout.flush()

        # Normalize
        trace /= maxrun
        return trace

    def __config__(self):
        """Configure network.

        Defines a Caffe network with the specified configuration. The network
        is configured from the `self.prototxt` and `self.caffemodel` files.

        Note:
          This functions creates a new instance of the network.
        """
        self.caffe_net = None
        self.caffe_net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

        # Construct transformation
        transformer_object = {'data': self.caffe_net.blobs['data'].data.shape}
        transformer = caffe.io.Transformer(transformer_object)

        # Apply
        transformer.set_mean('data', self.mean_image[0].mean(1).mean(1))
        transformer.set_raw_scale('data', 255)

    def __add_damage__(self, wanted_layer, std, with_bias=False):
        """Add Gaussian noise to the layers.

        Add Gaussian noise to the layers defined in `wanted_layers`. Use `std`
        to control the standard deviation of the Gaussian noise.

        Args:
          wanted_layer (List[str]): List containing layer names (eg [`conv1`]).
          std (float): Scale of the applied noise.
          with_bias (Optinal[bool]): Toggle use of bias parameters.
        """
        # Get the layer std
        net_std = self.__est_std__(wanted_layer, with_bias)

        for layer, dim in self.caffe_net.params.items():
            if layer in wanted_layer and std > 0.0:
                weight_shape = dim[0].data.shape
                bias_shape = dim[1].data.shape

                # Design damage (here white noise)
                weight_noise = np.random.normal(0.0, net_std*std, weight_shape)
                bias_noise = np.random.normal(0.0, net_std*std, bias_shape)

                # Apply
                self.caffe_net.params[layer][0].data[:] += weight_noise
                if with_bias:
                    self.caffe_net.params[layer][1].data[:] += bias_noise

    def __mean_image__(self):
        """Mean image initialization.

        Set the mean image of the data set.
        """
        mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()

        # Open file
        with open(self.meanfile, 'rb') as f:
            mean_blobproto_new.ParseFromString(f.read())
            self.mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)

    def __list_params__(self):
        """(Trained) params (weights and bias).

        Returns the weights and biases of the network.
        """
        weights = [(k, v[0].data.shape) for k, v in self.caffe_net.params.items()]
        bias = [(k, v[1].data.shape) for k, v in self.caffe_net.params.items()]
        return (weights, bias)

    def __list_blobs__(self):
        """Blob activations.

        Returns the list of layers of the network.
        """
        return [(k, v.data.shape) for k, v in self.caffe_net.blobs.items()]

    def __params_flatten__(self, wanted_layers):
        """(Trained) params of `wanted_layers`.

        Returns a flatten list of parameters learned.

        Args:
          wanted_alyers (List[str]): List of layer names.
        """
        weights = np.array([])
        bias = np.array([])

        for layer in self.caffe_net.params:
            if layer in wanted_layers:
                weights = np.concatenate((weights, self.caffe_net.params[layer][0].data.flatten()))
                bias = np.concatenate((bias, self.caffe_net.params[layer][1].data.flatten()))

        return (weights, bias)

    def __blob_flatten__(self, wanted_layers):
        """Blob activations of `wanted_layers`.

        Returns a flatten list of blob activations.

        Args:
          wanted_alyers (List[str]): List of layer names.
        """
        blob = list()
        for name in wanted_layers:
            blob.append(self.caffe_net.blobs[name].data[0][:].ravel())
        return blob

    def __est_mean__(self, wanted_layers, with_bias=False):
        """Estimate mean of `wanted_layers`.

        Returns the estimated mean value of the listed layers in
        `wanted_layers`. Use `with_bias` to include bias.

        Args:
          wanted_layers (List[str]): List of layer names (eg. ['conv1']).
          with_bias (Bool): Toggle bias.

        """
        weight, bias = self.__params_flatten__(wanted_layers)
        if with_bias is False:
            values = weight
        else:
            values = np.concatenate((weight, bias))
        return np.mean(values)

    def __est_std__(self, wanted_layers, with_bias=False):
        """Estimate std. of `wanted_layers`.

        Returns the estimated standard deviation of the listed layers in
        `wanted_layers`. Use `with_bias` to include bias.

        Args:
          wanted_layers (List[str]): List of layer names (eg. ['conv1']).
          with_bias (Bool): Toggle bias.

        """
        weight, bias = self.__params_flatten__(wanted_layers)
        if with_bias is False:
            values = weight
        else:
            values = np.concatenate((weight, bias))
        return np.std(values)

    def __gradients__(self, wanted_layers, with_bias=False):
        """Gradients of the `wanted_layers`.

        Returns the gradients calculated in an forward pass. Use `with_bias` to
        specify if to return gradients of bias paramters.

        Note:
          Gradients are only calculated if the `.prototext` file contains the
          additional `force_backwards=True` setting.

        Args:
          wanted_layers (List[str]): List containing layer names (eg [`conv1`]).
          with_bias (Bool): Toggle bias.

        """
        jacobi = np.array([])
        for k, v in self.caffe_net.params.items():
            if k in wanted_layers:
                if with_bias is False:
                    jacobi = np.concatenate((jacobi, self.caffe_net.params[k][0].diff[:].flatten()))
                else:
                    jacobi = np.concatenate((jacobi, self.caffe_net.params[k][0].diff[:].flatten(), self.caffe_net.params[k][1].diff[:].flatten()))
        return jacobi


if __name__ == "__main__" and __package__ is None:
    print 'This file contains the network class'

    # Define a network class
    net = Network(prototxt='/home/aaskov/caffe/examples/cifar10/cifar10_full.prototxt',
                  caffemodel='/home/aaskov/caffe/examples/cifar10/bkp_cifar10_full_converge_002/cifar10_full_iter_100000.caffemodel.h5',
                  meanfile='/home/aaskov/caffe/examples/cifar10/mean.binaryproto',
                  path_to_data='/home/aaskov/caffe/examples/cifar10/cifar10_train_lmdb/')

