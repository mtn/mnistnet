#ifndef __MACROS_H__
#define __MACROS_H__

#ifdef DEBUG
#include <stdio.h>

#define DEBUG_PRINT(x) printf x

#define PRINT_MATRIX(m) DEBUG_PRINT(("[\n\t")); \
    for (int __print_i = 0; __print_i < m.num_rows; __print_i++) { \
        for (int __print_j = 0; __print_j < m.num_cols; __print_j++) { \
            if (__print_j == m.num_cols - 1) { \
                DEBUG_PRINT(("%f\n", m.elem[__print_i * __print_j - 1])); \
                if (__print_i != m.num_rows - 1) { \
                    DEBUG_PRINT(("\t")); \
                } \
                continue; \
            } \
            DEBUG_PRINT(("%f, ", m.elem[__print_i * m.num_cols + __print_j])); \
        } \
    } \
    DEBUG_PRINT(("]\n"))

#else
#define DEBUG_PRINT(x)
#define PRINT_MATRIX(m)
#endif

#endif /* __MACROS_H__ */
