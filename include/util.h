#ifndef __UTIL_H__
#define __UTIL_H__


typedef struct {
    int num_rows;
    int num_cols;

    double* elem;
} Matrix;

// Initialize a matrix
Matrix* matrix_init(Matrix* m, int num_rows, int num_cols);
// Zero-initialize a matrix with a given size (like to np.zeros)
Matrix* matrix_init_zeros(Matrix* m, int num_rows, int num_cols);
// Initialize a matrix as a clone of another, copying the entire buffer
Matrix* matrix_init_from(Matrix* m, Matrix* from);

// Pass ownership of a buffer from one matrix to another (passing over dimensions as well)
// Takes ownership of the matrix from, invalidating the pointer
Matrix* matrix_into(Matrix* dest, Matrix* from);

// does the modification in place
void matrix_map_(Matrix* m, double (*map_fn)(double elem));

void matrix_init_buffer(Matrix* m, double (*init_fn)());

// Get an index to access a matrix row or column
int matrix_get_ind(Matrix* m, int row, int col);

// Free the memory owned my m, but not m itself (since a series of matrices could have
// been allocated contiguously at once)
void matrix_free(Matrix* m);


// Shuffle an array of integers in place (used in generating minibatches)
void shuffle_ints_(int* array, int len);


#endif /* __UTIL_H__ */
