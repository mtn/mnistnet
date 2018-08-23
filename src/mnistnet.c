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

    int sizes[3];
    sizes[0] = 784;
    sizes[1] = 30;
    sizes[2] = 10;

    Network* net = create_network(3, sizes);

    stochastic_gradient_descent(net, training_data, 30, 10, 3.0, NULL);

    free_network(net);
    free_mnist_data(training_data);
}
