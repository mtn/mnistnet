#include <stdlib.h>
#include <string.h>

#include "util.h"


#include "macros.h"
void matrix_init(Matrix* m, int num_rows, int num_cols) {
    m->num_rows = num_rows;
    m->num_cols= num_cols;
    m->elem = malloc(num_rows * num_cols * sizeof(double));
}

Matrix* matrix_map(Matrix* m, double (*map_fn)()) {
    Matrix* new = malloc(sizeof(Matrix));
    matrix_init(new, m->num_rows, m->num_cols);

    for (int i = 0; i < m->num_rows * m->num_cols; i++) {
        new->elem[i] = (*map_fn)();
    }

    return new;
}

void matrix_map_(Matrix* m, double (*map_fn)(double elem)) {
    for (int i = 0; i < m->num_rows * m->num_cols; i++) {
        m->elem[i] = (*map_fn)(m->elem[i]);
    }
}

void matrix_init_buffer(Matrix* m, double (*init_fn)()) {
    for (int i = 0; i < m->num_rows * m->num_cols; i++) {
        double ret = (*init_fn)();
        /* DEBUG_PRINT(("initfn return %f", ret)); */
        m->elem[i] = ret;
        /* DEBUG_PRINT(("assigned %f", ret)); */
    }
}

int matrix_get_ind(Matrix* m, int row, int col) {
    return row * m->num_cols + col;
}

void matrix_free(Matrix* m) {
    free(m->elem);
}
