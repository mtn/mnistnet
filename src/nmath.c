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

double sigmoid_prime(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

// Elementwise sigmoid function that modifies a vector in place
void matrix_sigmoid_(Matrix* m) {
    matrix_map_(m, &sigmoid);
}

void matrix_sigmoid_prime_(Matrix* m) {
    matrix_map_(m, &sigmoid_prime);
}

Matrix* matrix_scalar_multiply(Matrix* m, Matrix* b, double scalar) {
    m = matrix_init(m, b->num_rows, b->num_cols);

    for (int i = 0; i < m->num_rows * m->num_cols; i++) {
        m->elem[i] = b->elem[i] * scalar;
    }

    return m;
}

void check_multiplication_compatability(Matrix* a, Matrix* b) {
    // Should never be triggered, since matrix_multiply is only called internally
    if (a->num_cols != b->num_rows) {
        puts("Arguments to matrix multiply had incompatible shapes, exiting");

        DEBUG_PRINT(("\tShapes: (%d, %d) (%d, %d)\n", a->num_rows, a->num_cols,
                                                      b->num_rows, b->num_cols));
        DEBUG_PRINT(("\tRequired that %d == %d\n", a->num_cols, b->num_rows));

        exit(1);
    }
}

Matrix* matrix_multiply(Matrix* dest, Matrix* a, Matrix* b) {
    check_multiplication_compatability(a, b);
    DEBUG_PRINT(("Matrix multiply\n"));

    // Dest could be aliases with a or b, so we store pointers to their buffers
    double* a_buf = a->elem;
    double* b_buf = b->elem;

    Matrix* m = matrix_init(dest, a->num_rows, b->num_cols);

    for (int i = 0; i < a->num_rows; i++) {
        for (int j = 0; j < b->num_cols; j++) {
            m->elem[matrix_get_ind(m, i, j)] = 0;

            for (int k = 0; k < a->num_cols; k++) {
                m->elem[matrix_get_ind(m, i, j)] += a_buf[matrix_get_ind(a, i, k)]
                    * b_buf[matrix_get_ind(b, k, j)];
            }
        }
    }

    return m;
}

// Pointer aliasing is okay, because we keep the buffers
Matrix* matrix_dot_(Matrix* dest, Matrix* a, Matrix* b) {
    // If a or b is 1-D, equivalent to mapping scalar multiplication
    if (a->num_rows == 1 && a->num_cols == 1) {
        return matrix_scalar_multiply(dest, b, a->elem[0]);
    } else if (b->num_rows == 1 && b->num_cols == 1) {
        return matrix_scalar_multiply(dest, a, b->elem[0]);
    }

    return matrix_multiply(dest, a, b);
}

Matrix* matrix_dot(Matrix* a, Matrix* b) {
    return matrix_dot_(NULL, a, b);
}

int compute_broadcast_value(double* buf, int num_rows, int num_cols, int i, int j) {
    if (num_rows == 1) {
        if (num_cols == 1) {
            return buf[0];
        } else {
            return buf[j];
        }
    } else {
        if (num_cols == 1) {
            return buf[i];
        } else {
            return buf[num_cols * i + j];
        }
    }
}

void check_addition_compatibility(Matrix* a, Matrix* b) {
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
}

Matrix* matrix_add_(Matrix* a, Matrix* b, Matrix* dest) {
    check_addition_compatibility(a, b);

    // Since we know the matrix was broadcast-safe, we can just take the max
    int num_rows = MAX(a->num_rows, b->num_rows);
    int num_cols = MAX(a->num_cols, b->num_cols);
    //
    // Dest could be aliases with a or b, so we store pointers and values
    double* a_buf = a->elem;
    double* b_buf = b->elem;
    int a_num_rows = a->num_rows;
    int b_num_rows = b->num_rows;
    int a_num_cols = a->num_cols;
    int b_num_cols = b->num_cols;

    Matrix* m = matrix_init(dest, num_rows, num_cols);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            m->elem[matrix_get_ind(m, i, j)] = compute_broadcast_value(a_buf, a_num_rows, a_num_cols, i, j)
                + compute_broadcast_value(b_buf, b_num_rows, b_num_cols, i, j);
        }
    }

    return m;
}

Matrix* matrix_add(Matrix* a, Matrix* b) {
    return matrix_add_(a, b, NULL);
}

double negate(double x) {
    return -x;
}

// TODO see if there can be fewer allocations
// The problem is that broadcasting might make in-place addition/subtraction tricky
// For now, I'm just doing an additional allocation + deallocation
Matrix* matrix_subtract(Matrix* a, Matrix* b) {
    DEBUG_PRINT(("Subtracting matrices: \n"));
    PRINT_MATRIX(a);
    DEBUG_PRINT(("-\n"));
    PRINT_MATRIX(b);
    DEBUG_PRINT(("\n"));

    Matrix* neg= matrix_init_from(NULL, b);
    matrix_map_(neg, &negate);

    DEBUG_PRINT(("b negated: \n"));
    PRINT_MATRIX(neg);

    Matrix* res = matrix_add(a, neg);
    matrix_free(neg);
    free(neg);

    DEBUG_PRINT(("Result: \n"));
    PRINT_MATRIX((res));

    return res;
}

// TODO It would be nice to allow this to be done in place, since the memory layout of
// a matrix and its transpose are the same. The implementation is more complicated,
// so for now to avoid having to deallocate in the caller I'm just passing a flag
// that has the callee take ownership of the original matrix (by deallocating it)
Matrix* matrix_transpose(Matrix* m, bool take_ownership) {
    Matrix* trans = matrix_init(NULL, m->num_cols, m->num_rows);

    for (int i = 0; i < m->num_rows; i++) {
        for (int j = 0; j < m->num_cols; j++) {
            trans->elem[matrix_get_ind(trans, j, i)] = m->elem[matrix_get_ind(m, i, j)];
        }
    }

    if (take_ownership) {
        matrix_free(m);
        free(m);
    }

    return trans;
}
