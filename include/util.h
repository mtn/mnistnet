#ifndef __UTIL_H__
#define __UTIL_H__


typedef struct {
    int num_rows;
    int num_cols;

    double* elem;
} Matrix;

// Initialize a matrix
void matrix_init(Matrix* m, int num_rows, int num_cols);

// Initialize matrix values using an initialization function (eg. stdnormal)
// *_ does the modification in place, the normal one involves a new allocation
Matrix* matrix_map(Matrix* m, double (*map_fn)(double elem));
void matrix_map_(Matrix* m, double (*map_fn)(double elem));

void matrix_init_buffer(Matrix* m, double (*init_fn)());

// Get an index to access a matrix row or column
int matrix_get_ind(Matrix* m, int row, int col);

// Free the memory owned my m, but not m itself (since a series of matrices could have
// been allocated contiguously at once)
void matrix_free(Matrix* m);


#endif /* __UTIL_H__ */
