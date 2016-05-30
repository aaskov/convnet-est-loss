# -*- coding: utf-8 -*-
"""
convnet-est-loss
"""
import numpy as np
import argparse


#%%
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
                        
    parser.add_argument('--iterations', default=500, type=int,
                        help='Number of iterations to run.')
    
    args = parser.parse_args()
    

#%%
if __name__ == "__main__":
    run()