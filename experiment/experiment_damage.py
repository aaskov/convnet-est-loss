# -*- coding: utf-8 -*-
"""
convnet-est-loss
"""
import numpy as np
from os import sys, path
import argparse
caffe_dir = '/home/aaskov/caffe/'

#%%

def damage_range(x):
    if float(x) < 0.0:
        raise argparse.ArgumentTypeError("%r is not positive" % x)
    return x


def run():
    parser = argparse.ArgumentParser(
        description='Neural network damage experiment. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--proto', required=True, type=str,
                        help='Network prototxt.')
    parser.add_argument('--model', required=True, type=str,
                        help='Network caffemodel.')
    parser.add_argument('--meanfile', required=True, type=str,
                        help='Data mean file.')
    parser.add_argument('--data', required=True, type=str,
                        help='Data.')
    parser.add_argument('--layer', required=True, type=str,
                        help='Layer to apply damage to.')
    
    parser.add_argument('--damage', default=1.0, type=damage_range,
                        help='Applied damage range.')
    parser.add_argument('--step-num', default=10, type=int,
                        help='Number of steps in damage interval.')
    parser.add_argument('--iterations', default=100, type=int,
                        help='Number of iterations to run.')

    
    args = parser.parse_args()
    
    loss_list = list()
    for std in np.linspace(0.0, args.damage, args.step_num):
        # Fetch a network structure and apply damage
        net = Network(args.proto, args.model, args.meanfile, args.data)    
        net.__add_damage__(args.layer, std)
        
        # Forward to get loss
        top_1, top_5, acc, loss = net.forward(maxrun=args.iterations)
        loss_list.append(loss)

    # Store result
    save_obj(loss_list, 'experiment_damage_' + str(args.layer) + '_' + 
            str(args.damage) + '_step_' + str(args.step_num) + '_iter_' + 
            str(args.iterations))
    


#%%
if __name__ == '__main__' and __package__ is None:
    # Append parrent directory to sys path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from network import Network
    from input_output import save_obj
    
    # Setup Caffe
    sys.path.insert(0, caffe_dir + 'python')
    import caffe
    caffe.set_mode_cpu()
    
    # Run experiment
    run()
    
    