# Tests description

Run the script `make_test.py` within an environment with Tensor Fox installed. This script runs several routines of Tensor Fox:
	- It computes the CPD for a fixed tensor with different set of options
	- It computes the CPD of a high order tensor using the Tensor Train method, also with different set of options
	- It tests the condition number routines and its associated inequality
	- It computes the CPD of a high order sparse tensor, using sparse techniques  