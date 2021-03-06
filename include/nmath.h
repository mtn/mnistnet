#ifndef __NMATH_H__
#define __NMATH_H__

#include <stdbool.h>

#include "util.h"


// Return a standard-normal sampled number (deterministic per-run)
double stdnormal();

// Sigmoid function (*_ modifies in place)
void sigmoid_(Matrix* m);
double* sigmoid(Matrix* m);

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
Matrix* matrix_dot(Matrix* a, Matrix* b);
// Same but allows user to specify destination matrix
// Works and tested even under pointer aliasing
Matrix* matrix_dot_(Matrix* dest, Matrix* a, Matrix* b);
// Element-wise multiplication
Matrix* matrix_hadamard_product(Matrix* dest, Matrix* a, Matrix* b);

Matrix* matrix_add(Matrix* a, Matrix* b);
Matrix* matrix_subtract(Matrix* a, Matrix* b); // a - b

Matrix* matrix_transpose(Matrix* m);

// Apply the sigmoid function elemntwise, in place
void matrix_sigmoid_(Matrix* m);
void matrix_sigmoid_prime_(Matrix* m);

// Return the index of the maximum value in a matrix
int matrix_argmax(Matrix* m);


#endif /* __NMATH_H__ */
