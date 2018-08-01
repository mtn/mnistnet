#include <stdlib.h>
#include <string.h>

#include "util.h"

void doublearr_init(Vector* v, double (*init_fn)()) {
    for (int i = 0; i < v->len; i++) {
        v->elem[i] = (*init_fn)();
    }
}

// TODO check if this is used
Vector* new_vector(int veclen) {
    Vector* v = malloc(sizeof(Vector));

    v->len = veclen;
    v->elem = malloc(veclen * sizeof(double));

    return v;
}

// TODO check if this is used
Vector* into_vector(int veclen, double* buffer) {
    Vector* v = new_vector(veclen);

    memcpy(v->elem, buffer, veclen * sizeof(double));

    return v;
}

void init_vector(Vector* v, int veclen) {
    v->elem = malloc(veclen * sizeof(double));
    v->len = veclen;
}

// Frees the memory associated with the vector, but not the memory itself
void free_vector(Vector* v) {
    free(v->elem);
}
