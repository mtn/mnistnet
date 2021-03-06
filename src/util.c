#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "macros.h"


// Allocates a matrix if necessary, and claims the buffer if the matrix is already
// initialized. This prepares destination pointers to take a new buffer without leaking
// memory.
Matrix* alloc_or_own(Matrix* m) {
    if (m == NULL) {
        m = malloc(sizeof(Matrix));
        m->elem = NULL;
    } else if (m->elem != NULL) {
        matrix_free(m);
    }

    return m;
}

Matrix* matrix_init_shallow(int count) {
    if (count <= 0) {
        return NULL;
    }

    Matrix* ms = malloc(sizeof(Matrix) * count);

    // Ensure the elem pointer is null, so we know these matrices are uninitialized
    for (int i = 0; i < count; i++) {
        ms[i].elem = NULL;
    }

    return ms;
}

Matrix* matrix_init(Matrix* m, int num_rows, int num_cols) {
    m = alloc_or_own(m);

    m->num_rows = num_rows;
    m->num_cols= num_cols;
    m->elem = malloc(num_rows * num_cols * sizeof(double));

    return m;
}

// Creates (or updates) a matrix and initializes its values to zero
Matrix* matrix_init_zeros(Matrix* m, int num_rows, int num_cols) {
    m = alloc_or_own(m);

    m->num_rows = num_rows;
    m->num_cols = num_cols;
    m->elem = calloc(num_rows * num_cols, sizeof(double));

    return m;
}

// Initialize a matrix as a clone of another
// the caller is responsible for handling its buffer, if there was one
Matrix* matrix_init_from(Matrix* m, Matrix* from) {
    // If a matrix is passed in, it's buffer is assumed to be correctly size
    m = alloc_or_own(m);

    matrix_init(m, from->num_rows, from->num_cols);
    memcpy(m->elem, from->elem, from->num_rows * from->num_cols * sizeof(double));

    return m;
}

// Writes into dest
Matrix* matrix_into(Matrix* dest, Matrix* from) {
    dest = alloc_or_own(dest);

    dest->num_rows = from->num_rows;
    dest->num_cols = from->num_cols;
    dest->elem = from->elem;

    free(from);

    return dest;
}

void matrix_map_(Matrix* m, double (*map_fn)(double elem)) {
    for (int i = 0; i < m->num_rows * m->num_cols; i++) {
        m->elem[i] = (*map_fn)(m->elem[i]);
    }
}

void matrix_init_buffer(Matrix* m, double (*init_fn)()) {
    for (int i = 0; i < m->num_rows * m->num_cols; i++) {
        m->elem[i] = (*init_fn)();
    }
}

int matrix_get_ind(Matrix* m, int row, int col) {
    return row * m->num_cols + col;
}

// This doesn't free the pointer m, since it might not have been
// individually allocated
void matrix_free(Matrix* m) {
    free(m->elem);
}

// Source: https://stackoverflow.com/a/6127606/2608433
void shuffle_ints_(int* array, int len) {
    if (len > 1) {
        for (int i = 0; i < len - 1; i++) {
          int j = i + rand() / (RAND_MAX / (len - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}
