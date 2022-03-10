import os


print('Testing CPD with different options...')
os.system('python -m doctest simple_tensor_cpd_options.py')

print('\nTesting Tensor Train with different options...')
os.system('python -m doctest tensor_train_cpd.py')

print('\nTesting condition number...')
os.system('python -m doctest condition_number.py')

print('\nTesting CPD for sparse tensors...')
os.system('python -m doctest sparse_tensors.py')

print('\nFinished testing')