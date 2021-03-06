# -*- coding: utf-8 -*-
"""
convnet-est-loss
"""
import numpy as np
from os import sys, path
import argparse
caffe_dir = '/home/aaskov/caffe/'


def damage_range(x):
    if float(x) < 0.0:
        raise argparse.ArgumentTypeError("%r is not positive" % x)
    return float(x)


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
    parser.add_argument('--prefix', required=True, type=str,
                        help='Uniqe model name.')
    parser.add_argument('--damage', default=1.0, type=damage_range,
                        help='Applied damage range.')
    parser.add_argument('--step-num', default=10, type=int,
                        help='Number of steps in damage interval.')
    parser.add_argument('--iterations', default=100, type=int,
                        help='Number of iterations to run.')
    parser.add_argument('--repeat', default=0, type=int,
                        help='Number of repeated experiments to run.')
    args = parser.parse_args()
    
    # Object file
    obj = 'experiment_damage_' + str(args.layer) + '_' + str(args.damage)
    obj += '_step_' + str(args.step_num) + '_iter_' + str(args.iterations)
    obj += '_prefix_' + str(args.prefix)

    loss_list = list()
    for std in np.linspace(0.0, args.damage, args.step_num):
        _loss = list()
        for t in range(args.repeat + 1):
            # Fetch a network structure and apply damage
            net = Network(args.proto, args.model, args.meanfile, args.data)
            net.__add_damage__(args.layer, std)

            # Forward to get loss
            top_1, top_5, acc, loss = net.forward(maxrun=args.iterations)
            _loss.append(loss)

        loss_list.append(_loss)

    # Store result
    if path.exists(obj+'.pkl'):
        read_loss = load_obj(obj)
        combined = np.concatenate((read_loss, np.array(loss_list)), 1)
        save_obj(combined, obj)
    else:        
        save_obj(np.array(loss_list), obj)


if __name__ == '__main__' and __package__ is None:
    # Append parrent directory to sys path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from network import Network
    from input_output import save_obj, load_obj

    # Setup Caffe
    sys.path.insert(0, caffe_dir + 'python')
    import caffe
    caffe.set_mode_cpu()

    # Run experiment
    run()

