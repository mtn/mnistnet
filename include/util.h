#ifndef __UTIL_H__
#define __UTIL_H__

/// Return a standard-normal sampled number (deterministic per-run)
double stdnormal();

/// A initializizer for arrays of doubles
void doublearr_init(int len, double* arr, double (*init_fn)());

#endif /* __UTIL_H__ */
