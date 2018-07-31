#include <math.h>
#include <stdlib.h>

#include "util.h"

/// Return a standard-normal sampled double
/// Based on the Box-Muller Method
/// Source: stackoverflow.com/q/5817490/2608433
double stdnormal() {
    static double v, fac;
    static int phase = 0;
    double S, Z, U1, U2, u;

    if (phase) {
        Z = v * fac;
    } else {
        do {
            U1 = (double)rand() / RAND_MAX;
            U2 = (double)rand() / RAND_MAX;

            u = 2. * U1 - 1.;
            v = 2. * U2 - 1.;
            S = u * u + v * v;
        } while (S >= 1);

        fac = sqrt(-2. * log(S) / S);
        Z = u * fac;
    }

    phase = 1 - phase;

    return Z;
}

/// Initialize an array of doubles with the given initialization function
void doublearr_init(int len, double* arr, double (*init_fn)()) {
    for (int i = 0; i < len; i++) {
        arr[i] = (*init_fn)();
    }
}
