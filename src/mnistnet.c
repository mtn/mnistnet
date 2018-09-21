#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "network.h"
#include "macros.h"
#include "mnist.h"
#include "nmath.h"
#include "util.h"


int main() {
    // Load in the first 50000 examples as training data
    // The remaining data is loaded as a validation set at &training_data[1]
    MnistData* training_data = load_data("data/train-labels-idx1-ubyte",
                                         "data/train-images-idx3-ubyte",
                                         50000);

    // Load the full test set (0 loads everything)
    MnistData* test_data = load_data("data/t10k-labels-idx1-ubyte",
                                     "data/t10k-images-idx3-ubyte",
                                     0);

    int sizes[3];
    sizes[0] = 784;
    sizes[1] = 30;
    sizes[2] = 10;

    Network* net = create_network(3, sizes);

    stochastic_gradient_descent(net, training_data, 10, 10, 3.0, test_data);

    free_network(net);
    free_mnist_data(training_data);
    free_mnist_data(&training_data[1]);
    free(training_data);
    free_mnist_data(test_data);
    free(test_data);
}
