# MPI-based implementation of Communication-Efficient Stratified Stochastic Gradient Descent for Distributed Matrix Completion

## Compiling the code

'''
run make
'''

Usage: bin/dsgdpar [options] <Matrix file>
 Options:
	-i number of iterations (epochs), default: 10
	-f number of latent factors, default: 16 NOTE: give as a srtring e.g "30"
	-l regularization factor (lambda, default=0.0075)
	-e learning rate (eps, default=0.0015)
	-c communication type:
		0: Block-wise (DSGD)
		1: P2P
		3: P2P with Hold and Combine
	-s Strata schedule type:		0: RING_FIXED_SEED (default) picks a seed randomly at first epoch and sticks to it in all epochs
		1: RING_RANDOM_SEED picks a different seed randomly at each epoch
	-p Partition file, if not provided random partitioning of rows and cols is used. 
	-d Force random partition of column even if a partition file is provided
	-h Print this help message

