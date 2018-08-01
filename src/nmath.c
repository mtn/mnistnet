#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "macros.h"
#include "util.h"


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


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Elementwise sigmoid function that modifies a vector in place
void matrix_sigmoid_(Matrix* m) {
    matrix_map_(m, &sigmoid);
}

// Maps the sigmoid function over a matrix (returns a new vector)
/* Matrix* sigmoid(Matrix* m) { */
/*     Matrix* copy = */ 
/*     double* copy = malloc(veclen * sizeof(Matrix)); */
/*     memcpy(copy, vec, veclen * sizeof(double)); */

/*     sigmoid_(copy, veclen); */
/*     return copy; */
/* } */

Matrix* matrix_multiply(Matrix* a, Matrix* b) {
    if (a->num_cols != b->num_rows) {
        puts("Arguments to matrix multiply had incompatible shapes, exiting");

        DEBUG_PRINT(("\tShapes: (%d, %d) (%d, %d)\n", a->num_rows, a->num_cols,
                                                      b->num_rows, b->num_rows));
        DEBUG_PRINT(("\tRequired that %d == %d\n", a->num_cols, b->num_rows));

        exit(1);
    }

    Matrix* m = malloc(sizeof(Matrix));
    matrix_init(m, a->num_rows, b->num_cols);

    for (int i = 0; i < a->num_rows; i++) {
        for (int j = 0; j < b->num_cols; j++) {
            m->elem[matrix_get_ind(m, i, j)] = 0;

            for (int k = 0; k < a->num_cols; k++) {
                m->elem[matrix_get_ind(m, i, j)] += a->elem[matrix_get_ind(a, i, k)]
                    * b->elem[matrix_get_ind(b, k, j)];
            }
        }
    }

    return m;
}

Matrix* matrix_add(Matrix* a, Matrix* b) {
    if (a->num_cols != b->num_cols || a->num_rows != b->num_rows) {
        puts("Arguments to matrix add had incompatible shapes, exiting");

        DEBUG_PRINT(("\tShapes: (%d, %d) (%d, %d)\n", a->num_rows, a->num_cols,
                                                      b->num_rows, b->num_rows));
        DEBUG_PRINT(("\tRequired that %d == %d, %d == %d\n", a->num_cols, b->num_cols,
                                                             a->num_rows, b->num_rows));

        exit(1);
    }

    Matrix* m = malloc(sizeof(Matrix));
    matrix_init(m, a->num_rows, a->num_cols);

    for (int i = 0; i < a->num_rows; i++) {
        for (int j = 0; j < a->num_cols; j++) {
            m->elem[matrix_get_ind(m, i, j)] = a->elem[matrix_get_ind(a, i, j)]
                + b->elem[matrix_get_ind(b, i, j)];
        }
    }

    return m;
}
