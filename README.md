# TensorCodes

Given a tensor $T$ we want to approximate it by a tensor $S$ with low rank $r$. If the dimension are high this is a challenging problem. In the case od dimension $2$, the tensors are matrices and this problem is solved: we can just compute the SVD of $T$ and construct $S$ as a sum of $r$ rank $1$ matrices. It is proved that this is the best approximation of $T$ with rank $r$. Already in dimension $3$ (i.e., $T \in V_1 \otimes V_2 \otimes V_3$, where each $V_i$ is a real vector space) this result fails and is possible that $T$ doesn't have a best rank $r$ approximation, but infinite ones.

The module *gauss_newton* is a high performance package of routines made specifically to compute these approximations in dimension $3$. The name comes from the fact that we are using a Damped Gauss-Newton method to make these approximations. 

**References:**<br />

Tensors: Geometry and Applications. Author: J. M. Landsberg<br />
Most tensor problems are NP-hard. Authors: Christopher Hillar, Lek-Heng Lim<br />
https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm<br />
https://pdfs.semanticscholar.org/1deb/b5581b538e75dbee8fd07bda36382baea977.pdf
