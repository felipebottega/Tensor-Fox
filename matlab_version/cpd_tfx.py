import numpy as np
import os
import sys
from sys import argv
import scipy.io as spio


"""
This function prepares all variables saved through Matlab to be used in Tensor Fox. Then in calls the 
Tensor Fox CPD function. If save_results = 1, them all results are saved in the files factors.mat, 
T_approx.mat and output.mat (see Tensor Fox documentation for more about these outputs).
"""


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


# Load filenames.
tensor_path = argv[1]
R = int(argv[2])
options_path = argv[3]
save_results = int(argv[4])
path_tfx = argv[5]

# Add path to Tensor Fox and load the module.
sys.path.append(path_tfx)
import prepare_files as pf
import TensorFox as tfx

# Load array and options.
T = pf.loadarray(tensor_path)
options = pf.loadstructure(options_path)

# Compute CPD.
factors, T_approx, final_outputs = tfx.cpd(T, R, options)

if save_results:
    # Save factors matrices.
    L = len(factors)
    factors_dict = {}
    for l in range(L):
        factors_dict['fac' + str(l)] = factors[l]
    path = os.path.join("outputs", "factors.mat")
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
    path = os.path.join("outputs", "output.mat")
    spio.savemat(path, output)
