#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>

#include "network.h"
#include "macros.h"
#include "nmath.h"
#include "util.h"


void test_network_init() {
    int* sizes = malloc(sizeof(int) * 3);

    sizes[0] = 1;
    sizes[1] = 2;
    sizes[2] = 3;

    Network* net = create_network(3, sizes);

    assert(net->sizes[0] == 1);
    assert(net->sizes[1] == 2);
    assert(net->sizes[2] == 3);

    free_network(net);
}

void test_matrix_multiplication() {
    Matrix* m1 = malloc(sizeof(Matrix));
    matrix_init(m1, 2, 3);
    m1->elem[0] = 1;
    m1->elem[1] = 2;
    m1->elem[2] = 3;
    m1->elem[3] = 4;
    m1->elem[4] = 5;
    m1->elem[5] = 6;

    assert(m1 != NULL);
    assert(m1->num_rows == 2);
    assert(m1->num_cols == 3);

    Matrix* m2 = malloc(sizeof(Matrix));
    matrix_init(m2, 3, 2);
    m2->elem[0] = 7;
    m2->elem[1] = 8;
    m2->elem[2] = 9;
    m2->elem[3] = 10;
    m2->elem[4] = 11;
    m2->elem[5] = 12;

    assert(m2 != NULL);
    assert(m2->num_rows == 3);
    assert(m2->num_cols == 2);

    Matrix* m3 = matrix_multiply(m1, m2);

    assert(m3->num_rows == 2);
    assert(m3->num_cols == 2);
    assert(m3->elem[0] == 58);
    assert(m3->elem[1] == 64);
    assert(m3->elem[2] == 139);
    assert(m3->elem[3] == 154);
}

void run(void (*test_fn)()) {
    Dl_info info;

    int result = dladdr((const void*)test_fn, &info);
    if (result == 0) {
        puts("Failed to find test function name, aborting!");
        exit(1);
    }

    test_fn();

    printf("Test %s succeeded!\n", info.dli_sname);
}

int main () {
    puts("Running tests");

    run(&test_network_init);
    run(&test_matrix_multiplication);
}
