# Tensor Fox

Tensor Fox is a high performance package of multilinear algebra and tensor routines, with focus on the Canonical Polyadic Decomposition (CPD).

## Table of Contents
* [ :fox_face: Motivation](#motivation)
* [ :fox_face: Getting Started](#started)
* [ :fox_face: Performance](#performance)
* [ :fox_face: Structure of Tensor Fox](#structure-of-tensor-fox)
* [ :fox_face: Author](#author)
* [ :fox_face: License](#license)
* [ :fox_face: References](#references)

## :fox_face: Motivation

 Multidimensional data structures are common these days, and extracting useful information from them is crucial for several applications. For bidimensional data structures (i.e., matrices), one can rely in decompositions such as the [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) for instance. There are two  possible generalizations of the SVD for multidimensional arrays of higher order: the *multilinear singular value decomposition* (MLSVD) and the *canonical polyadic decomposition* (CPD). The former can be seen as a [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) of higher order and is useful for dimensionality reduction, whereas the latter is useful to detect latent variables. Computing the MLSVD is just a matter of computing several SVD's, but the CPD is a challenging problem.

![alt text](https://github.com/felipebottega/Tensor-Fox/blob/master/readme_files/tensor-intuition.png)

 Determinating the [rank](https://en.wikipedia.org/wiki/Tensor_rank_decomposition#Tensor_rank) is a NP-hard problem, so the best option is to rely on heuristics, guessing and estimates. Although the value of the rank is a hard task, once we have its value or a reasonable estimate, computing an approximated CPD is a polynomial task. There are several implementations of algorithms to compute a CPD, but most of them relies on the *alternating least squares* (ALS) algorithm, which is cheap to compute but has severe convergence issues. Algorithms like the *damped Gauss-Newton* (dGN) are more robust but in general are much more costly. Tensor Fox is a CPD solver for Python (with Numpy and Numba as backend) which manages to use the dGN algorithm in a cheap way, being robust while also being competitive with ALS in terms of speed. Furthermore, Tensor Fox offers several additional multilinear algebra routines in the context of tensors. 

## :fox_face: Getting Started

In order to have everything working properly, all files of Tensor Fox must be in the same folder of your program. At the moment we are only offering the module files, so there is no install procedure to follow, just download the modules and import them locally. To be able to use Tensor Fox properly you will need the following packages already installed on your computer:

    numpy
    pandas
    scipy
    sklearn
    matplotlib
    numba

Make sure Numba and Numpy updated. Additionaly, make sure you are using a nice version of BLAS. That is all! Tensor Fox is read to go! 

Let's start importing Tensor Fox and other necessary modules for now.

![alt text](https://github.com/felipebottega/Tensor-Fox/blob/master/readme_files/ipynb1.png)

### Creating Tensors and Getting Information 

Let's create a little tensor **T** just to see how Tensor Fox works at its basics. For third order tensors (3D arrays) I use the convention that **T**[ijk] refers to the *i*-th row, *j*-column and *k*-slice (frontal slice) of **T**. For instance, consider the tensor defined above (the frontal slices of **T** are showed).

![alt_text](https://github.com/felipebottega/Tensor-Fox/blob/master/readme_files/formula1.png)

Since Numpy's convention is different from ours with regard to third order tensors. This convention may be irrelevant when using the routines of Tensor Fox, but since I build all the modules thinking this way, it is fair that this point is made explicitly. The function **showtens** prints a third order tensor with this particular convention and print tensors of higher order just as Numpy would print. Below we show both conventions with an example of third order tensor. 

![alt_text](https://github.com/felipebottega/Tensor-Fox/blob/master/readme_files/ipynb2.png)

### Computing the CPD

Now let's turn to the most important tool of Tensor Fox, the computation of the CPD. We can compute the corresponding CPD simply calling the function **cpd** with the tensor and the rank as arguments. This is just the tip of the iceberg of Tensor Fox, to know more check out the [tutorial](https://github.com/felipebottega/Tensor-Fox/tree/master/tutorial) and the [examples of applications](https://github.com/felipebottega/Tensor-Fox/tree/master/examples). 

![alt_text](https://github.com/felipebottega/Tensor-Fox/blob/master/readme_files/ipynb3.png)

## :fox_face: Performance

In the following we compare the performances of Tensor Fox and other known tensor packages: Tensorlab, Tensor Toolbox and Tensorly. Our first benchmark consists in measuring the effort for the other solvers to obtain a solution close to the Tensor Fox default. We compute the CPD of four fixed tensors:

 1) *Swimmer*: This tensor was constructed in [this](https://github.com/felipebottega/Tensor-Fox/blob/master/tutorial/6-first_problem.ipynb) tutorial lesson. It is a set of 256 images of dimensions 32 x 32 representing a swimmer. Each image contains a torso (the invariant part) of 12 pixels in the center and four limbs of 6 pixels that can be in one of 4 positions. We proposed to use a rank R = 50 tensor to approximate it.
 
 2) *Handwritten digits*: This is a classic tensor in machine learning, it is the [MNIST](http://yann.lecun.com/exdb/mnist/}) database of handwritten digits. Each slice is a image of dimensions 20 x 20 of a handwritten digit. Also, each 500 consecutive slices correspond to the same digit, so the first 500 slices correspond to the digit 0, the slices 501 to 1000 correspond to the digit 1, and so on. We choose R = 150 as a good rank to construct the approximating CPD to this tensor. This tensor is also used [here](https://github.com/felipebottega/Tensor-Fox/blob/master/examples/handwritten_digit.ipynb), where we present a tensor learning technique for the problem of classification. 
 
 3) *Border rank*: The phenomenon of [border rank](https://en.wikipedia.org/wiki/Tensor_rank_decomposition#Border_rank) can make the CPD computation a challenging problem. The article [1] has a great discussion on this subject. In the same article they show a tensor of rank 3 and border rank 2. We choose to compute a CPD of rank R = 2 to see how the algorithms behaves when we try to approximate a problematic tensor by tensor with low rank. In theory it is possible to have arbitrarily good approximations. 
			
 4) *Matrix multiplication*: Matrix multiplication between square n x n matrices can be seen as a tensor of shape n² x n² x n². Since [Strassen](https://en.wikipedia.org/wiki/Strassen_algorithm) it is known that these multiplications can be made with less operations. For the purpose of testing we choose the small value n = 5 and the rank R = 92. However note that this is probably not the exact rank of the tensor (it is lower), so this test is about a strict low rank approximation of a difficult tensor.
 
**PS**: Tensor Fox relies on Numba, which makes compilations [just in time](http://numba.pydata.org/) (JIT). This means that the first run will compile the functions and this take some time. From the second time on there is no more compilations and the program should take much less time to finish the computations. Any measurement of performance should be made after the compilations are done. Additionally, be warned that changing the order of the tensor, the method of initialization or the method of the inner algorithm also triggers more compilations. Before running Tensor Fox for real we recommend to run it on a small tensor, just to make the compilations.  
 
 For each tensor we make 100 runs of the Tensor Fox's CPD and keep the best solution (smallest error). Now let ALG be any other algorithm. First we set the tolerance option to a very small value so the algorithm don't stop because of tolerance conditions. After that we set the maximum number of iterations to *maxiter* = 5 and run ALG with these options 100 times. We only accept the solutions with relative error smaller that *error + error/100*, where *error* is the relative error obtained with Tensor Fox. Between all accepted solutions we select that one with the smallest running time. If no solution is found with these number of iterations, we increase it to *maxiter* = 10 and repeat. We try the values *maxiter* = 5, 10, 50, 100, 150, 200, 250, ..., 900, 950, 1000, until there is an accepted solution. Otherwise we consider that ALG failed. Note that these procedures favors all algorithms against Tensor Fox since we are trying to use the small possible number of iterations for them. We used random initilization in all tests. The results are showed below.
 
 ![alt_text](https://github.com/felipebottega/Tensor-Fox/blob/master/readme_files/benchmarks1.png)
 
 The are using the following abbreviations for the algorithms:
 
  * NLS: Tensorlab NLS without refinement
  * NLSr: Tensorlab NLS with refinement
  * TlabALS: Tensorlab ALS withou refinement
  * TlabALSr: Tensorlab ALS with refinement
  * MINF: Tensorlab MINF without refinement
  * MINFr: Tensorlab MINF with refinement
  * TlyALS: Tensorly ALS
  * OPT: Tensor Toolbox OPT (with 'lbfgs' algorithm)
  
  Now we consider another round of tests, by instead of fixed tensors we use a family of tensors. More precisely, we consider random tensors of shape *n* x *n* x ... x *n* and rank R = 5, where the entries of each factor matrix are drawn from the normal distribution (mean 0 and variance 1). First we consider fourth tensors with shape *n* x *n* x *n* x *n*, for *n* = 10, 20, 30, 40, 50, 60, 70, 80. Since the Tensorlab's NLS performed very well in the previous texts, we start with this algorithm, making 20 computations for each dimension *n* and avering the errors and time. After that we run the other algorithm adjusting their tolerance in order to match the NLS results. 
  
 In all tests we tried to choose the options in order to speed up the algorithms without losing accuracy. For example, we noticed that it was unnecessary to use compression, detection of structure and refinement for the NLS algorithm. These routines are very expensive and didn't bring much extra precision, so they were disabled in order to make the NLS computations faster. Similarly we used the initialization 'svd' for Tensorly because it proved to be faster than 'random', and we used the algorithm 'lbfgs' for Tensor Toolbox OPT. Finally, for Tensor Fox we just decreased its tolerance in order to match the precision given by the NLS algorithm. The results are showed below. 
  
![alt_text](https://github.com/felipebottega/Tensor-Fox/blob/master/readme_files/benchmarks2.png)

 Next, we make the same procudure but this time we fixed *n* to *n* = 10 and increased the order, from order 3 to 8. These last tests shows an important aspect of Tensor Fox: it avoids the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality), whereas the other algorithms still suffers from that. We consider random rank-5 tensors of shape 10 x 10 x 10, them 10 x 10 x 10 x 10, up to tensors of order 8, i.e., with shape 10 x 10 x 10 x 10 x 10 x 10 x 10 x 10, with the same distribution as before. The routine to generate theses kind of tensors can be found [here](https://github.com/felipebottega/Tensor-Fox/blob/master/tests/gen_rand_tensor.py).
 
![alt_text](https://github.com/felipebottega/Tensor-Fox/blob/master/readme_files/benchmarks3.png)

## :fox_face: Structure of Tensor Fox

In this section we summarize all the features Tensor Fox has to offer. As already mentioned, computing the CPD is the main goal of Tensor Fox, but in order to accomplish this mission several 'sub-goals' had to be overcome first. Many of these sub-goals ended up being important routines of multilinear algebra. Besides that, during the development of this project several convenience routines were added, such as statistics analysis of tensors, rank estimation, automated plotting with CPD information, and many more. Below we present the modules of Tensor Fox and gives a brief description of their main functions.

|**TensorFox**|  |
|---|---|
| cpd| computes the CPD of a tensor **T** with rank *R*. |
| rank| estimates the rank of a tensor.|
| stats| given a tensor **T** and a rank *R*, this fucntions computes some statistics regarding the CPD computation. |
| foxit| does the same job as the *cpd* function but at the end it prints and plots relevant information. |
   
|**Auxiliar**|  |
|---|---|
| tens2matlab| given a tensor, this function creates a Matlab file containing the tensor and its dimensions. |
| sort_dims| given a tensor, this function sort its dimensions in descending order and returns the sorted tensor. |
| rank1| given the factors of a CPD, this function converts them into a matrix, which is the first frontal slice of the tensor in coordinates obtained by this rank-1 term. |
   
| **Compression**|  |
|---|---|
| mlsvd| computes the MLSVD of a tensor. |
| clean_compression| truncates the MLSVD. |
   
| **Conversion**|  |
|---|---|
| cpd2tens| converts the factor matrices to tensor in coordinate format. |
| unfold| given a tensor and a choice of a mode, this function computes the unfolding of the tensor with respect of that mode.  |
| foldback| given a matrix representing a unfolding of some mode and the dimensions of the original tensor, this function retrieves the original tensor from its unfolding. |
| normalize| normalize the columns of the factors to have unit column norm and introduce a central tensor with the scaling factors. |
| denormalize| given the normalized factors together with a central tensor, this function retrives the non-normalized factors. |
| equalize| make the vectors of each mode to have the same norm. |
   
| **Critical**| |
|---|---|
|   | this module responsible for the most costly parts of Tensor Fox (basically it is a module of boring loops) |

| **Display**|  |
|---|---|
| infotens| display several informations about a given tensor. |
| test_tensors| a function made specifically to test different models against different tensors. It is very useful when one is facing difficult tensors and needs to tune the parameters accordingly. |
   
| **GaussNewton**|   |
|---|---|
| dGN| [damped Gauss-Newton](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm) function adapated for the tensor problem.. |
| cg| [conjugate gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method) function specifically made for the tensor problem. |
| lsmr| [LSMR](http://web.stanford.edu/group/SOL/software/lsmr/) function adapated for the tensor problem. |
| regularization| computes the [Tikhonov matrix](https://en.wikipedia.org/wiki/Tikhonov_regularization) for the inner algorithm. |
| precond| computes the [preconditioner matrix](https://en.wikipedia.org/wiki/Preconditioner) for the inner algorithm.  |
   
| **Initialization**|  |
|---|---|
| starting_point| main function to generates the starting point. There are four possible methods of initialization, 'random', 'smart_random', 'smart', or you can provide your own starting point. |
| find_factor| if the user introduce constraints to the entries of the solution, a projection is made at each step of the dGN. This projection is based on three parameters, where the least clear is the *factor* parameter. This function helps the user to find the best factor for the starting point. For more information, see this notebook. |
   
| **MultilinearAlgebra**| |
|---|---|
| multilin_mult| performs the multilinear multiplication. |
| multirank_approx| given a tensor **T** and a prescribed multirank (R1, ..., Rm), this function tries to find the (almost) best approximation of **T** with multirank (R1, ..., Rm). |
| kronecker| computes the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) between two matrices. |
| khatri_rao| computes the [Khatri-Rao product](https://en.wikipedia.org/wiki/Kronecker_product#Khatri%E2%80%93Rao_product) between two matrices. |
| hadamard| computes the [Hadamar product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) between two matrices. |

## :fox_face: Author

* Felipe B. Diniz: https://github.com/felipebottega
* Contact email: felipebottega@gmail.com
* Linkedin: https://www.linkedin.com/in/felipe-diniz-4a212163/?locale=en_US
* Kaggle: https://www.kaggle.com/felipebottega

## :fox_face: License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the [LICENSE.md](https://github.com/felipebottega/Tensor-Fox/blob/master/LICENSE) file for details.
    

## :fox_face: References

 1) V. de Silva, and L.-H. Lim, *Tensor Rank and the Ill-Posedness of the Best Low-Rank Approximation Problem*, SIAM Journal on Matrix Analysis and Applications, 30 (2008), pp. 1084-1127. 
 2) P. Comon, X. Luciani, and A. L. F. de Almeida, *Tensor Decompositions, Alternating Least Squares and other Tales*, Journal of Chemometrics, Wiley, 2009.   
 3) T. G. Kolda and B. W. Bader, *Tensor Decompositions and Applications*, SIAM Review, 51:3, in press (2009).   
 4) J. M. Landsberg, *Tensors: Geometry and Applications*, AMS, Providence, RI, 2012.   
 5) B. Savas, and Lars Eldén, *Handwritten Digit Classification Using Higher Order Singular Value Decomposition*, Pattern Recognition Society, vol. 40, no. 3, pp. 993-1003, 2007.
 6) C. J. Hillar, and L.-H. Lim. *Most tensor problems are NP-hard*, Journal of the ACM, 60(6):45:1-45:39, November 2013. ISSN 0004-5411. doi: 10.1145/2512329.
 7) A. Shashua, and T. Hazan, *Non-negative Tensor Factorization with Applications to Statistics and Computer Vision*, Proceedings of the 22nd International Conference on Machine Learning (ICML), 22 (2005), pp. 792-799.
 8) S. Rabanser, O. Shchur, and S. Günnemann, *Introduction to Tensor Decompositions and their Applications in Machine Learning*, arXiv:1711.10781v1 (2017). 
 9) A. H. Phan, P. Tichavsky, and A. Cichoki, *Low Complexity Damped Gauss-Newton Algorithm for CANDECOMP/PARAFAC*, SIAM Journal on Matrix Analysis and Applications, 34 (1), 126-147 (2013).
 10) L. De Lathauwer, B. De Moor, and J. Vandewalle, *A Multilinear Singular Value Decomposition*, SIAM J. Matrix Anal. Appl., 21 (2000), pp. 1253-1278.
 11) https://www.tensorlab.net/
 12) http://www.sandia.gov/~tgkolda/TensorToolbox/
 13) https://github.com/tensorly/
