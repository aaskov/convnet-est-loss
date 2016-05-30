# -*- coding: utf-8 -*-
"""
convnet-est-loss
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import lmdb
import time

caffe_dir = '/home/aaskov/caffe/'
os.chdir(caffe_dir)
sys.path.insert(0, './python')
import caffe


#%%

class Network(object):	
    """Example text text text text.

    Text text text text text text text text text text text text text text text
    text text text text.

    Note:
        Text `self` text ``Args`` section.

    Args:
        param1 (str): Text of `param1`.
        param2 (Optional[int]): Text of `param2`.
        param3 (List[str]): Text of `param3`.

    """
    def __init__(self, prototxt, caffemodel, meanfile, path_to_data):
        """Example text text text text.

        Text text text text text text text text text text text text text text
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        self.prototxt = prototxt
        self.caffemodel = caffemodel
        self.meanfile = meanfile
        self.path_to_data = path_to_data
        
        # Setup and configure
        self.__mean_image__()
        self.__config__()


    def forward(self, maxrun=10000, k=5, post=1000):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        lmdb_env = lmdb.open(self.path_to_data)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe.proto.caffe_pb2.Datum()
        
        # Loop stats
        count = 0
        accuracy = 0.0
        top1_error = 0.0
        top5_error = 0.0
        
        for key, value in lmdb_cursor:
            t = time.time()        
            
            # Extract the image from the LMDB dataset
            datum.ParseFromString(value)
            label = int(datum.label)
            input_image = caffe.io.datum_to_array(datum)
    
            # Insert into the network
            self.caffe_net.blobs['data'].data[...] = input_image
            net_out = self.caffe_net.forward()
            
            # Evaluate
            predict_label = int(net_out['prob'][0].argmax(axis=0))
            output_prob = self.caffe_net.blobs['prob'].data[0]
            top_inds = output_prob.argsort()[::-1][:k]
            
            # Stats
            iscorrect = predict_label == label
            accuracy += (1 if iscorrect else 0)
            top1_error += (0 if label == top_inds[0] else 1)
            top5_error += (0 if label in top_inds[:k] else 1)
            
            count += 1
            if count % post == 0:
                text = 'Running %i. Acc: %0.5f. Time: %0.5f'
                print text % (count, accuracy/count, post*(time.time()-t))
            if count > maxrun: break
        
        # Normalize
        accuracy /= count
        top1_error /= count
        top5_error /= count        
        
        return((accuracy, top1_error, top5_error))


    def hessian(self, wanted_layers, maxrun=10000, with_bias=False):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        lmdb_env = lmdb.open(self.path_to_data)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe.proto.caffe_pb2.Datum()
    
        # For pre-config
        diff = self.__gradients__(self.caffe_net, wanted_layers, with_bias)
        trace = np.zeros_like(diff)
    
        # Loop over images
        post = 1000
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
            self.caffe_net.blobs['prob'].diff[0][label] = 1  # Suggested from caffe-users google forum
            self._backward = self.caffe_net.backward(**{self.caffe_net.outputs[0]: self.caffe_net.blobs['prob'].diff})
    
            # Calculate the derivatices and store the approx Hessian
            diff = self.__gradients__(wanted_layers, with_bias)
            trace += np.power(diff, 2)
    
            count += 1
            if count > maxrun: break
            if count % post == 0: 
                print 'Running %i. Time: %0.5f' % (count, post*(time.time()-t))
                sys.stdout.flush()
    
        # Normalize
        trace /= maxrun
    
        return trace


    def __config__(self):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        self.caffe_net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)
        
        # Construct transformation
        transformer_object = {'data': self.caffe_net.blobs['data'].data.shape}
        transformer = caffe.io.Transformer(transformer_object)
        
        # Apply
        transformer.set_mean('data', self.mean_image[0].mean(1).mean(1))
        transformer.set_raw_scale('data', 255)


    def __mean_image__(self):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
        
        # Open file
        with open(self.meanfile, 'rb') as f:
            mean_blobproto_new.ParseFromString(f.read())
            self.mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)


    def __list_params__(self):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        weights = [(k, v[0].data.shape) for k, v in self.caffe_net.params.items()]
        bias = [(k, v[1].data.shape) for k, v in self.caffe_net.params.items()]
        return (weights, bias)


    def __list_blobs__(self):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        return [(k, v.data.shape) for k, v in self.caffe_net.blobs.items()]


    def __params_flatten__(self, wanted_layers):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        weights = np.array([])
        bias = np.array([])
    
        for layer in self.caffe_net.params:
            if layer in wanted_layers:
                weights = np.concatenate((weights, self.caffe_net.params[layer][0].data.flatten()))
                bias = np.concatenate((bias, self.caffe_net.params[layer][1].data.flatten()))
    
        return (weights, bias)


    def __blob_flatten__(self, wanted_layers):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        blob = list()
        for name in wanted_layers:
            blob.append(self.caffe_net.blobs[name].data[0][:].ravel())
        return blob


    def __est_mean__(self, wanted_layers, with_bias=False):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        weight, bias = self.__params_flatten__(wanted_layers)
        if with_bias is False:
            values = weight
        else:
            values = np.concatenate((weight, bias))
        return np.mean(values)


    def _est_std__(self, wanted_layers, with_bias=False):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        weight, bias = self.__params_flatten__(wanted_layers)
        if with_bias is False:
            values = weight
        else:
            values = np.concatenate((weight, bias))
        return np.std(values)


    def __gradients__(self, wanted_layers, with_bias=False):
        """Example text text text text.

        Text text text text text text text text text text text text text text 
        text text text text.
        
        Note:
          Text `self` text ``Args`` section.
        
        Args:
          param1 (str): Text of `param1`.
          param2 (Optional[int]): Text of `param2`.
          param3 (List[str]): Text of `param3`.
        
        """
        jacobi = np.array([])
        for k, v in self.caffe_net.params.items():
            if k in wanted_layers:
                if with_bias is False:
                    jacobi = np.concatenate((jacobi, self.caffe_net.params[k][0].diff[:].flatten()))
                else:
                    jacobi = np.concatenate((jacobi, self.caffe_net.params[k][0].diff[:].flatten(), self.caffe_net.params[k][1].diff[:].flatten()))
        return jacobi


if __name__ == "__main__":
	print 'This file contains the network class'
 
        model = '002'
        prototxt = '/home/aaskov/caffe/examples/cifar10/cifar10_full_with_force_backward.prototxt'
        caffemodel = '/home/aaskov/caffe/examples/cifar10/bkp_cifar10_full_converge_'+model+'/cifar10_full_iter_100000.caffemodel.h5'
        mean_file = '/home/aaskov/caffe/examples/cifar10/mean.binaryproto'
        path_to_data = '/home/aaskov/caffe/examples/cifar10/cifar10_train_lmdb/' 
        
        net = Network(prototxt, caffemodel, mean_file, path_to_data)