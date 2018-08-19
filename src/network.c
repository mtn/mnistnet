#include <stdlib.h>
#include <string.h>

#include "load_mnist.h"
#include "network.h"
#include "macros.h"
#include "nmath.h"
#include "util.h"


// Initialize bias vectors
// Assumes that net->sizes and net->num_layers are properly initialized
void init_biases(Network* net) {
    DEBUG_PRINT(("\nInitializing bias vectors:\n"));

    // Each row has a bias matrix (vector)
    net->biases = malloc((net->num_layers - 1) * sizeof(Matrix));
    for (int i = 0; i < net->num_layers - 1; i++) {
        // Initialize a column vector with {# nodes in next row} nodes
        matrix_init(&net->biases[i], net->sizes[i + 1], 1);
        matrix_init_buffer(&net->biases[i], &stdnormal);

        PRINT_MATRIX((&net->biases[i]));
    }
}

// Initialize weight vectors
// Assumes that net->sizes and net->num_layers are properly initialized
void init_weights(Network* net) {
    DEBUG_PRINT(("\nInitializing weights:\n"));

    // Each row has a weights matrix
    net->weights = malloc((net->num_layers - 1) * sizeof(Matrix));
    for (int i = 1; i < net->num_layers; i++) {
        DEBUG_PRINT(("Layer %d-%d:\n", i, i + 1));

        matrix_init(&net->weights[i - 1], net->sizes[i], net->sizes[i - 1]);
        matrix_init_buffer(&net->weights[i - 1], &stdnormal);

        PRINT_MATRIX((&net->weights[i - 1]));
    }

}

Network* create_network(int num_layers, int sizes[]) {
    Network* net = malloc(sizeof(Network));

    net->num_layers = num_layers;

    net->sizes = sizes;

    init_biases(net);
    init_weights(net);

    return net;
}

void free_network(Network* net) {
    free(net->sizes);

    // Free the memory associated with each bias matrix
    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_free(&net->biases[i]);
    }
    free(net->biases);

    // Free the memory associated with each weight matrix
    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_free(&net->weights[i]);
    }
    // Free the memory associated with each weight matrix
    free(net->weights);

    free(net);
}

#include <stdio.h>
// Feedforward
Matrix* feed_forward(Network* net, Matrix* inp) {
    for (int i = 0; i < net->num_layers - 1; i++) {
        Matrix* wa = matrix_dot(&net->weights[i], inp);
        DEBUG_PRINT(("w * a: \n"));
        PRINT_MATRIX(wa);

        matrix_free(inp);
        free(inp);

        Matrix* wab = matrix_add(wa, &net->biases[i]);
        DEBUG_PRINT(("wa + b\n"));
        PRINT_MATRIX(wab);

        matrix_free(wa);
        free(wa);

        matrix_sigmoid_(wab);

        inp = wab;
    }

    return inp;
}

// Mini-batch stochastic gradient descent
void stochastic_gradient_descent(int mini_batch_size, int num_epoch) {

}
