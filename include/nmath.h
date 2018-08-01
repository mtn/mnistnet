#ifndef __NMATH_H__
#define __NMATH_H__

#include "util.h"


// Return a standard-normal sampled number (deterministic per-run)
double stdnormal();

// Sigmoid function (*_ modifies in place)
void sigmoid_(Matrix* m);
double* sigmoid(Matrix* m);

// Matrix multiplication (returns a freshly allocated matrix)
// Panics on incorrect dimensions
Matrix* matrix_multiply(Matrix* a, Matrix* b);


#endif /* __NMATH_H__ */
