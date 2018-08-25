#ifndef __MACROS_H__
#define __MACROS_H__


#define MAX(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define MIN(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _b : _a; })

#ifdef DEBUG
#include <stdio.h>

#define DEBUG_PRINT(x) printf x

#define PRINT_MATRIX(m) DEBUG_PRINT(("[\n\t")); \
    for (int __print_matrix_i = 0; __print_matrix_i < m->num_rows; __print_matrix_i++) { \
        for (int __print_matrix_j = 0; __print_matrix_j < m->num_cols; __print_matrix_j++) { \
            DEBUG_PRINT(("%f, ", m->elem[__print_matrix_i * (m)->num_cols + __print_matrix_j])); \
            if (__print_matrix_j == m->num_cols - 1) { \
                DEBUG_PRINT(("\n")); \
                if (__print_matrix_i != m->num_rows - 1) { \
                    DEBUG_PRINT(("\t")); \
                } \
                continue; \
            } \
        } \
    } \
    DEBUG_PRINT(("]\n"))

#define PRINT_MNIST_IMG(m) \
    for (int __print_mnist_img_i = 0; __print_mnist_img_i < 28; __print_mnist_img_i++) {\
        for (int __print_mnist_img_k = 0; __print_mnist_img_k < 28; __print_mnist_img_k++) { \
            DEBUG_PRINT((m.pixels[__print_mnist_img_i * 28 + __print_mnist_img_k] > 0 ? "1" : "0")); \
        } \
        DEBUG_PRINT(("\n")); \
    }

#define PRINT_DATAHEAD(d) \
    DEBUG_PRINT(("\nMnist Data Head: \n\n")); \
    for (int __print_datahead_i = 0; __print_datahead_i < MIN(5, d->count); __print_datahead_i++) { \
        PRINT_MNIST_IMG(d->images[__print_datahead_i]); \
        DEBUG_PRINT(("Label: %d\n\n",d->labels[__print_datahead_i])); \
    }

#else

#define DEBUG_PRINT(x)
#define PRINT_MATRIX(m)
#define PRINT_DATAHEAD(d)

#endif /* DEBUG */

#define P_MATRIX(m) printf("[\n\t"); \
    for (int __print_matrix_i = 0; __print_matrix_i < m->num_rows; __print_matrix_i++) { \
        for (int __print_matrix_j = 0; __print_matrix_j < m->num_cols; __print_matrix_j++) { \
            printf("%f, ", m->elem[__print_matrix_i * (m)->num_cols + __print_matrix_j]); \
            if (__print_matrix_j == m->num_cols - 1) { \
                printf("\n"); \
                if (__print_matrix_i != m->num_rows - 1) { \
                    printf("\t"); \
                } \
                continue; \
            } \
        } \
    } \
    printf("]\n")

#endif /* __MACROS_H__ */
