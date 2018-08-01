#ifndef __NMATH_H__
#define __NMATH_H__


// Return a standard-normal sampled number (deterministic per-run)
double stdnormal();

// Sigmoid function (*_ modifies in place)
void sigmoid_(double* vec, int veclen);
double* sigmoid(double* vec, int veclen);


#endif /* __NMATH_H__ */
