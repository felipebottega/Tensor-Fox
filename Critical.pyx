import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from cython cimport wraparound, boundscheck

 
@boundscheck(False)
@wraparound(False)
def update_datat(double[:] data, double[:] datat, long[:] datat_id):
    # 'Bridge' between Python and the C function _update_datat.
    cdef int size = data.shape[0]    
    datat = _update_datat(data, datat, datat_id, size)    
    return datat


@boundscheck(False)
@wraparound(False)
cdef _update_datat(double[:] data, double[:] datat, long[:] datat_id, int size): 
    # Updates datat given the current values of data.
    cdef int i
    with nogil:
        for i in prange(size, schedule="guided"):
            datat[i] = data[datat_id[i]]        
    return datat


@boundscheck(False)
@wraparound(False)
def matvec(double[:] z, double[:] data, long[:] col, double[:] x, int r, int m, int n, int p):
    # 'Bridge' between Python and the C function matvec.
    z = _matvec(z, data, col, x, r, m, n, p)
    return z


@boundscheck(False)
@wraparound(False)
def rmatvec(double[:] z, double[:] datat, long[:] colt, double[:] x, int r, int m, int n, int p):
    # 'Bridge' between Python and the C function _rmatvec.
    z = _rmatvec(z, datat, colt, x, r, m, n, p)
    return z
                 

@boundscheck(False)
@wraparound(False)
cdef _matvec(double[:] z, double[:] data, long[:] col, double[:] x, int r, int m, int n, int p):
    # Computes A*x, where A is given by data and col.
    cdef int num_per_row = 4*r
    cdef int num_rows = m*n*p   
    cdef int low, high, i, j
    cdef double zi
         
    with nogil:       
        for i in prange(num_rows, schedule="guided"):
            low = num_per_row*i
            high = num_per_row*(i+1)
            
            zi = 0.0
            for j in range(low, high):
                zi = zi + data[j]*x[col[j]]
            z[i] = zi        
            
    return z


@boundscheck(False)
@wraparound(False)
cdef _rmatvec(double[:] z, double[:] datat, long[:] colt, double[:] x, int r, int m, int n, int p):    
    # Computes A^T*x, where A^T is given by datat and colt.
    cdef int size1 = m*n*p
    cdef int size2 = n*p
    cdef int size3 = m*p
    cdef int size4 = m*n
    cdef int const1 = r*m
    cdef int const2 = r*n
    cdef int const3 = r*p
    cdef int const = r*m*n*p
    cdef int low, high, i, j
    cdef double zi 
             
    with nogil:        
        for i in prange(r + const1 + const2 + const3):
            if (i >= 0) and (i < r):
                low = size1*i
                high = size1*(i+1)
         
            elif (i >= r) and (i < r + const1):
                low = const + size2*(i - r)
                high = const + size2*(i + 1 - r)
         
            elif (i >= r + const1) and (i < r + const1 + const2):
                low = 2*const + size3*(i - r - const1)
                high = 2*const + size3*(i + 1 - r - const1)
         
            else:
                low = 3*const + size4*(i - r - const1 - const2)
                high = 3*const + size4*(i + 1 - r - const1 - const2)
             
            zi = 0.0
            for j in range(low, high):
                zi = zi + datat[j]*x[colt[j]]
            
            z[i] = zi
            
    return z


@boundscheck(False)
@wraparound(False)
cdef _rmatvec2(double[:] z, double[:] datat, long[:] colt, double[:] x, int r, int m, int n, int p):    
    # Computes A^T*x, where A^T is given by datat and colt.
    cdef int size1 = m*n*p
    cdef int size2 = n*p
    cdef int size3 = m*p
    cdef int size4 = m*n
    cdef int const1 = r*m
    cdef int const2 = r*n
    cdef int const3 = r*p
    cdef int const = r*m*n*p
    cdef int low, high, i, j
    cdef double zi 
              
    with nogil:        
        for i in prange(0, r, schedule="guided"):
            low = size1*i
            high = size1*(i+1)
            zi = 0.0
            for j in range(low, high):
                zi = zi + datat[j]*x[colt[j]]            
            z[i] = zi
         
        for i in prange(r, r + const1, schedule="guided"):
            low = const + size2*(i - r)
            high = const + size2*(i + 1 - r)
            zi = 0.0
            for j in range(low, high):
                zi = zi + datat[j]*x[colt[j]]            
            z[i] = zi
         
        for i in prange(r + const1, r + const1 + const2, schedule="guided"):
            low = 2*const + size3*(i - r - const1)
            high = 2*const + size3*(i + 1 - r - const1)
            zi = 0.0
            for j in range(low, high):
                zi = zi + datat[j]*x[colt[j]]            
            z[i] = zi
         
        for i in prange(r + const1 + const2, r + const1 + const2 + const3, schedule="guided"):
            low = 3*const + size4*(i - r - const1 - const2)
            high = 3*const + size4*(i + 1 - r - const1 - const2)
            zi = 0.0
            for j in range(low, high):
                zi = zi + datat[j]*x[colt[j]]            
            z[i] = zi             
            
    return z
