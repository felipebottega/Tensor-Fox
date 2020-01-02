import numpy as np
import os
import time
import sys
from sys import argv
import scipy.io as spio


def get_filename(tensor):
    """
    Given a path 'tensor' to the .mat file where the tensor is stored, this function extracts
    the filename from the path. For example, if tensor = home/usr/folder/tensor_xyz.mat, then
    get_filename(tensor) outputs 'tensor_xyz'.
    """
    
    # Remove the '.mat' or '.npy' from the name.
    tensor = tensor[:-4]
    
    # Keeps the string, from right to left, until the symbol '/' is found.
    pieces = tensor.split('/')
    tensor_name = pieces[-1]
    return tensor_name


def stop_msg(output_stop):
    """
    This function saves the stopping messages of the main stage and refinement stage of Tensor Fox.
    """

    stop = ['0', '2']

    # stop_main message
    if output_stop[0] == 0:
        stop[0] = 'Main stage: Relative error is small enough.'
    if output_stop[0] == 1:
        stop[0] = 'Main stage: Steps are small enough.'
    if output_stop[0] == 2:
        stop[0] = 'Main stage: Improvement in the relative error is small enough.'
    if output_stop[0] == 3:
        stop[0] = 'Main stage: Gradient is small enough.'
    if output_stop[0] == 4:
        stop[0] = 'Main stage: Average of the last k = 1 + int(maxiter/10) relative errors is small enough.'
    if output_stop[0] == 5:
        stop[0] = 'Main stage: Limit of iterations was reached.'
    if output_stop[0] == 6:
        stop[0] = 'Main stage: dGN diverged.'

    # stop_refine message
    if output_stop[1] == 0:
        stop[1] = 'Refinement stage: Relative error is small enough.'
    if output_stop[1] == 1:
        stop[1] = 'Refinement stage: Steps are small enough.'
    if output_stop[1] == 2:
        stop[1] = 'Refinement stage: Improvement in the relative error is small enough.'
    if output_stop[1] == 3:
        stop[1] = 'Refinement stage: Gradient is small enough.'
    if output_stop[1] == 4:
        stop[1] = 'Refinement stage: Average of the last k = 1 + int(maxiter/10) relative errors is small enough.'
    if output_stop[1] == 5:
        stop[1] = 'Refinement stage: Limit of iterations was reached.'
    if output_stop[1] == 6:
        stop[1] = 'Refinement stage: dGN diverged.'
    if output_stop[1] == 7:
        stop[1] = 'Refinement stage: No refinement was performed.'

    return stop


# Initialize lists.
errors = []
timings = []

# Load filenames.
tensor_paths = argv[1]
ranks_path = argv[2]
options_paths = argv[3]
save_results = int(argv[4])
path_tfx = argv[5]

# Add path to Tensor Fox and load the module.
sys.path.append(path_tfx)
import prepare_files as pf
import TensorFox as tfx

# Create lists with tensor filenames, their corresponding ranks and options.
with open(tensor_paths) as f:
    tensor_list = f.readlines()
tensor_list = [x.strip() for x in tensor_list]

with open(ranks_path) as f:
    rank_list_txt = f.readlines()[0]
    rank_list_txt = rank_list_txt.split('\n')[0]
    with open(rank_list_txt) as ff:
        rank_list = ff.readlines()
    rank_list = [x.strip() for x in rank_list]

with open(options_paths) as f:
    options_list = f.readlines()
options_list = [x.strip() for x in options_list]

# Compute CPDs.
print('Tensor name            Relative error            Timing')
for tensor, rank, opt in zip(tensor_list, rank_list, options_list):
    # Load inputs to CPD.
    T = pf.loadarray(tensor)
    R = int(rank)
    options = pf.loadstructure(opt)
    
    # Compute CPD.
    start = time.time()
    factors, T_approx, final_outputs = tfx.cpd(T, R, options)
    end = time.time()
    
    # Save list with relative errors and timings.
    errors.append(final_outputs.rel_error)
    timings.append(end - start)
    
    # Display tensor name, relative error and timing to compute the current CPD.
    tensor_name = get_filename(tensor)
    error = '{:^30.5e}'.format(errors[-1])
    timing = '{:.3f}'.format(timings[-1]) + ' sec'
    print(tensor_name, error, timing)
    
    if save_results:
        # Save factors matrices.
        L = len(factors)
        factors_dict = {}
        for l in range(L):
            factors_dict['fac' + str(l)] = factors[l]
        path = os.path.join("outputs", tensor_name + "-factors.mat")
        spio.savemat(path, factors_dict)

        # Save some output information.
        output = {}
        output['num_steps'] = final_outputs.num_steps
        output['rel_error'] = final_outputs.rel_error
        output['accuracy'] = max(0, 100*(1 - final_outputs.rel_error))
        output['options'] = final_outputs.options
        if L == 3:
            output['step_sizes'] = final_outputs.step_sizes
            output['errors'] = final_outputs.errors
            output['improv'] = final_outputs.improv
            output['gradients'] = final_outputs.gradients
            output['stop_msg'] = stop_msg(final_outputs.stop)
        path = os.path.join("outputs", tensor_name + "-output.mat")
        spio.savemat(path, output)
        
    spio.savemat('errors.mat', {'errors': errors})
    spio.savemat('timings.mat', {'timings': timings})
