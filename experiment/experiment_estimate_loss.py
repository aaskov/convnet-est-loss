# -*- coding: utf-8 -*-
"""
convnet-est-loss
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from os import sys, path
caffe_dir = '/home/aaskov/caffe/'


def damage_interval(x):
    if float(x) < 0.0:
        raise argparse.ArgumentTypeError("%r is not positive" % float(x))
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
    parser.add_argument('--damage', default=1.0, type=damage_interval,
                        help='Applied damage range.')
    parser.add_argument('--step-num', default=10, type=int,
                        help='Number of steps in damage interval.')
    parser.add_argument('--iterations', default=100, type=int,
                        help='Number of iterations to run.')
    parser.add_argument('--repeat', default=0, type=int,
                        help='Number of repeated damage experiments.')
    args = parser.parse_args()

    # Object file
    obj = 'experiment_damage_' + str(args.layer) + '_' + str(args.damage)
    obj += '_step_' + str(args.step_num) + '_iter_' + str(args.iterations)

    # If file not exists
    if not path.isfile(obj + '.pkl'):

        # Calculate damage
        loss_list = list()
        for std in np.linspace(0.0, args.damage, args.step_num):
            _loss = list()
            for repeat in range(args.repeat + 1):
                # Fetch a network structure and apply damage
                net = Network(args.proto, args.model, args.meanfile, args.data)
                net.__add_damage__(args.layer, std)

                # Forward to get loss
                top_1, top_5, acc, loss = net.forward(maxrun=args.iterations)
                _loss.append(loss)

            loss_list.append(_loss)

        # Save data
        save_obj(np.array(loss_list), obj)

    # If file exists
    else:
        # Load data
        loss_list = load_obj(obj)

    # Calculate hessian
    net = Network(args.proto, args.model, args.meanfile, args.data)
    # Get the layer std. and damage interval
    net_std = net.__est_std__(args.layer)
    damage_range = np.linspace(0.0, args.damage, args.step_num)
    damage_span = net_std*damage_range

    # Get the Hessian and calculate the expected loss
    trace = net.get_hessian(args.layer, maxrun=np.min((args.iterations, 1e5)))
    expected_loss = 0.5*(damage_span**2)*np.sum(trace)+np.mean(loss_list, 1)[0]

    # Make plot?
    plt.rcParams['figure.figsize'] = (6, 6)
    plt.figure(1)
    plt.plot(damage_range, np.mean(loss_list, 1), 'k-', label='Measured loss')
    plt.plot(damage_range, expected_loss, 'k--', label='Est. loss')
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('Std. damage')
    plt.ylabel('Loss')
    plt.title('Estimated loss on layer ' + str(args.layer))
    plt.savefig(obj + '.pdf', format='pdf')


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

