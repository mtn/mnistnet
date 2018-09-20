#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "network.h"
#include "macros.h"
#include "mnist.h"
#include "nmath.h"
#include "util.h"


int main() {
    /* puts("Hello world!"); */

    // Load in the first 50000 examples as training data
    // The remaining data is loaded as a validation set
    MnistData* training_data = load_data("data/train-labels-idx1-ubyte",
                                         "data/train-images-idx3-ubyte",
                                         50000);

    // Load the full test set (0 loads everything)
    MnistData* test_data = load_data("data/t10k-labels-idx1-ubyte",
                                     "data/t10k-images-idx3-ubyte",
                                     0);

    /* PRINT_DATAHEAD((training_data)); */
    /* PRINT_DATAHEAD((test_data)); */

    // Print training data
    puts("Training data:");
    for (int i = 0; i < 50000; i++) {
        printf("%d\n", training_data[0].labels[i]);
    }


    puts("Validation data:");
    // Print validation data
    for (int i = 0; i < 10000; i++) {
        printf("%d\n", training_data[1].labels[i]);
    }

    // Print test data
    puts("Test data:");
    for (int i = 0; i < 10000; i++) {
        printf("%d\n", test_data[0].labels[i]);
    }

    int sizes[3];
    sizes[0] = 784;
    sizes[1] = 30;
    sizes[2] = 10;

    Network* net = create_network(3, sizes);

    puts("Network init params:");
    puts("Bias shapes:");
    for (int i = 0; i < net->num_layers - 1; i++) {
        printf("(%d, %d)\n", net->biases[i].num_rows, net->biases[i].num_cols);
    }
    puts("Weight shapes:");
    for (int i = 0; i < net->num_layers - 1; i++) {
        printf("(%d, %d)\n", net->weights[i].num_rows, net->weights[i].num_cols);
    }

    printf("Num Layers: %d\n", net->num_layers);
    puts("Nodes in each layer:");
    for (int i = 0; i < net->num_layers; i++) {
        printf("%d\n", net->sizes[i]);
    }

    puts("All the biases:");
    for (int i = 0; i < net->num_layers - 1; i++) {
        printf("Biases [%d]\n", i);

        for (int j = 0; j < net->biases[i].num_rows * net->biases[i].num_cols; j++) {
            printf("%.1f\n", net->biases[i].elem[j]);
        }
    }

    puts("All the weights:");
    for (int i = 0; i < net->num_layers - 1; i++) {
        printf("Weights [%d]\n", i);

        for (int j = 0; j < net->weights[i].num_rows * net->weights[i].num_cols; j++) {
            printf("%.1f\n", net->weights[i].elem[j]);
        }
    }

    exit(0);

    // Just one epoch, large minibatches
    stochastic_gradient_descent(net, training_data, 1, 10, 3.0, test_data);

    free_network(net);
    free_mnist_data(training_data);
    free(training_data);
    free_mnist_data(test_data);
    free(test_data);
}
