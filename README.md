## Communication-Efficient Stratified Stochastic Gradient Descent for Distributed Matrix Completion

CESSGD is an MPI implementation for scalable matrix-completion-based collaborative filtering.
Written in C, the code provides a communication-efficient parallelization of the *stratified stochastic gradient descent* algorithm proposed by [Gemulla et al., 2011](https://dl.acm.org/doi/abs/10.1145/2020408.2020426)


## Compiling, running and usage info

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

**Matrix file format**  
The sparse rating matrix should be stored as a list of (row, col, nnz) tuples in [Matrix Market](https://math.nist.gov/MatrixMarket/formats.html) format (.mtx).

**Partition file format**  
The partition file may contain M rows or M+N rows, where M and N are respectively the rows and cols of the input matrix. Each row has a single integer value between 0 and K-1, where K is the number of parts/processors, indicating that the corresponding input matrix row/col is assigned to this part/processor.  

**Example usage on an MPI cluster**  

```
$mpirun -n  512 bin/cessgd -i 1000 -f 20 -c 3 -s 1  <path to rating matrix file>
```

## Bug reports  
Please report any bugs to abubaker.nf@gmail.com

## Citing



```
@article{cessgd2022,
author = "Nabil Abubaker and M. Ozan Karsavuran and Cevdet Aykanat",
title = "{Scaling Stratified Stochastic Gradient Descent for Distributed Matrix Completion}",
year = "2022",
month = "3",
url = "https://www.techrxiv.org/articles/preprint/Scaling_Stratified_Stochastic_Gradient_Descent_for_Distributed_Matrix_Completion/19350536",
doi = "10.36227/techrxiv.19350536.v1"
}
```

## LICENCE
Please see the LICENCE file for more information
