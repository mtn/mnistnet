#ifndef __NMATH_H__
#define __NMATH_H__

#include "util.h"


// Return a standard-normal sampled number (deterministic per-run)
double stdnormal();

// Sigmoid function (*_ modifies in place)
void sigmoid_(Matrix* m);
double* sigmoid(Matrix* m);

/* Matrix* matrix_multiply(Matrix* a, Matrix* b); */
// Behaves mostly the same as numpy's np.dot. Specificially (paraphrasing from the docs):
//
// If both a and b are 1-D arrays, it is inner product of vectors
// (this isn't handled explicitly, since it's just a case of matrix multiplication)
// If both a and b are 2-D arrays, it is matrix multiplication
// If either a or b is 0-D (scalar), it is equivalent to multiply
//
// Does not implement
// If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
// If a is an N-D array and b is an M-D array (where M>=2),
// it is a sum product over the last axis of a and the second-to-last axis of b:
Matrix* matrix_multiply(Matrix* a, Matrix* b);
Matrix* matrix_dot(Matrix* a, Matrix* b);

Matrix* matrix_add(Matrix* a, Matrix* b);

// Apply the sigmoid function elemntwise, in place
void matrix_sigmoid_(Matrix* m);


#endif /* __NMATH_H__ */
