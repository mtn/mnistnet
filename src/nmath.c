#include <stdbool.h>
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

// Wrapper around scalar multiplication, for mapping purposes
double scalar_multiply(double a, double b) {
    return a * b;
}

// Elementwise sigmoid function that modifies a vector in place
void matrix_sigmoid_(Matrix* m) {
    matrix_map_(m, &sigmoid);
}

Matrix* matrix_scalar_multiply(Matrix* m, Matrix* b, double scalar) {
    matrix_init(m, b->num_rows, b->num_cols);

    for (int i = 0; i < m->num_rows * m->num_cols; i++) {
        m->elem[i] = b->elem[i] * scalar;
    }

    return m;
}

Matrix* matrix_multiply(Matrix* a, Matrix* b) {
    // Should never be triggered, since matrix_multiply is only called internally
    if (a->num_cols != b->num_rows) {
        puts("Arguments to matrix multiply had incompatible shapes, exiting");

        DEBUG_PRINT(("\tShapes: (%d, %d) (%d, %d)\n", a->num_rows, a->num_cols,
                                                      b->num_rows, b->num_cols));
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

Matrix* matrix_dot(Matrix* a, Matrix* b) {
    Matrix* m = malloc(sizeof(Matrix));

    // If a or b is 1-D, equivalent to mapping scalar multiplication
    if (a->num_rows == 1 && a->num_cols == 1) {
        return matrix_scalar_multiply(m, b, a->elem[0]);
    } else if (b->num_rows == 1 && b->num_cols == 1) {
        return matrix_scalar_multiply(m, a, b->elem[0]);
    }

    return matrix_multiply(a, b);

    exit(1);
}

int compute_broadcast_index(Matrix* m, int i, int j) {
    if (m->num_rows == 1) {
        if (m->num_cols == 1) {
            return m->elem[0];
        } else {
            return m->elem[j - 1];
        }
    } else {
        if (m->num_cols == 1) {
            return m->elem[i - 1];
        } else {
            return m->elem[matrix_get_ind(m, i, j)];
        }
    }
}

Matrix* matrix_add(Matrix* a, Matrix* b) {
    bool cols_compatible = a->num_cols == b->num_cols || a->num_cols == 1 || b->num_cols == 1;
    bool rows_compatible = a->num_rows == b->num_rows || a->num_rows == 1 || b->num_rows == 1;
    if (!cols_compatible || !rows_compatible) {
        puts("Arguments to matrix add had incompatible shapes, exiting");

        DEBUG_PRINT(("\tShapes: (%d, %d) (%d, %d)\n", a->num_rows, a->num_cols,
                                                      b->num_rows, b->num_cols));
        DEBUG_PRINT(("\tRequired that %d == %d, %d == %d\n", a->num_cols, b->num_cols,
                                                             a->num_rows, b->num_rows));

        exit(1);
    }

    // Since we know the matrix was broadcast-safe, we can just take the max
    int num_rows = MAX(a->num_rows, b->num_rows);
    int num_cols = MAX(a->num_cols, b->num_cols);

    Matrix* m = malloc(sizeof(Matrix));
    matrix_init(m, num_rows, num_cols);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            m->elem[matrix_get_ind(m, i, j)] = compute_broadcast_index(a, i, j)
                + compute_broadcast_index(b, i, j);
        }
    }

    return m;
}
