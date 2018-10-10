# TensorCodes

Given a tensor **T** we want to approximate it by a tensor **S** with low rank *r*. If the dimensions are high this is a challenging problem. In the case of tensors of order 2, they are just matrices and this problem is already solved: compute the SVD of **T** and, from this decomposition, we can construct **S** as a sum of *r* rank 1 matrices. It is proved that this is the best approximation of **T** with rank *r*. Already for tensors of order 3 (i.e., **T** is in **V_1 ⊗ V_2 ⊗ V_3**, where each **V_i** is a real (or complex) vector space) this result fails and is possible that **T** doesn't have a best rank **r** approximation.

The module *Gauss_Newton* is a high performance package of routines made specifically to compute these approximations for tensors of order 3. The name comes from the fact that we are using a Damped Gauss-Newton method to make these approximations. 

**References:**<br />

Tensors: Geometry and Applications. Author: J. M. Landsberg<br />
Most tensor problems are NP-hard. Authors: Christopher Hillar, Lek-Heng Lim<br />
https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm<br />
https://pdfs.semanticscholar.org/1deb/b5581b538e75dbee8fd07bda36382baea977.pdf
