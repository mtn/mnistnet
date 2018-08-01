#include <stdlib.h>
#include <string.h>
#include <math.h>

// Return a standard-normal sampled double
// Based on the Box-Muller Method
// Source: stackoverflow.com/q/5817490/2608433
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


// Elementwise sigmoid function that modifies a vector in place
void sigmoid_(double* vec, int veclen) {
    for (int i = 0; i < veclen; i++) {
        vec[i] = 1.0 / (1.0 + exp(-vec[i]));
    }
}

// Maps the sigmoid function over a vector (returns a new vector)
double* sigmoid(double* vec, int veclen) {
    double* copy = malloc(veclen * sizeof(double));
    memcpy(copy, vec, veclen * sizeof(double));

    sigmoid_(copy, veclen);
    return copy;
}
