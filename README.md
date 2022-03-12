## Communication-Efficient Stratified Stochastic Gradient Descent for Distributed Matrix Completion

CESSGD is an MPI implementation for scalable matrix-completion-based collaborative filtering.
Written in C, the code provides a communication-efficient parallelization of the *stratified stochastic gradient descent* algorithm proposed by [Gemulla et al., 2011](https://dl.acm.org/doi/abs/10.1145/2020408.2020426)

For more info:  
- [Nabil Abubaker](mailto:nabil.abubaker@bilkent.edu.tr)
- [Ozan Karsavuran](mailto:ozankar@gmail.com)
- [Cevdet Aykanat](mailto:aykanat@cs.bilkent.edu.tr)


### Compiling, running and usage info

```
$make
$bin/cessgd --help

Usage: bin/cessgd [options] <Matrix file>  
 Options:  
	-i maximum number of iterations (epochs), default: 10  
	-f number of latent factors, default: 16 NOTE: give as a srtring e.g., "30"  
	-l regularization factor (lambda, default=0.0075)  
	-e learning rate (eps, default=0.0015)  
	-c communication type:  
		0: Block-wise (DSGD)  
		1: P2P  
		3: P2P with Hold and Combine  
	-s Strata schedule type:  
		0: RING_FIXED_SEED (default) picks a seed randomly at first epoch and sticks to it in all epochs  
		1: RING_RANDOM_SEED picks a different seed randomly at each epoch  
	-p Partition file, if not provided then random partitioning of rows and cols is used.   
	-d Force random partition of column even if a partition file is provided  
	-h Print this help message  
```
**Example usage on an MPI cluster**  

```
$mpirun -n  512 bin/cessgd -i 1000 -f 20 -c 3 -s 1  <path to rating matrix file>
```

### LICENCE
Please see the LICENCE file for more information
