#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "network.h"
#include "macros.h"
#include "mnist.h"
#include "nmath.h"
#include "util.h"


int main() {
    puts("Hello world!");

    MnistData* training_data = load_data("data/train-labels-idx1-ubyte",
                                         "data/train-images-idx3-ubyte");

    PRINT_DATAHEAD((training_data));

    free_mnist_data(training_data);


    int sizes[3];
    sizes[0] = 1;
    sizes[1] = 2;
    sizes[2] = 3;

    Network* net = create_network(3, sizes);

    free_network(net);

    Matrix* m1 = matrix_init(NULL, 2, 2);
    m1->elem[0] = 1;
    m1->elem[1] = 3;
    m1->elem[2] = 5;
    m1->elem[3] = 7;

    printf("Matrix");
    PRINT_MATRIX(m1);
}
