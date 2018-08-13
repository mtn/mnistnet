#include <stdlib.h>
#include <string.h>

#include "network.h"
#include "macros.h"
#include "nmath.h"
#include "util.h"


#include <stdio.h>
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

#include <stdio.h>
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
        printf("Matrix dot: (%d, %d) x (%d, %d)\n", *&net->weights[i].num_rows, *&net->weights[i].num_cols, inp->num_rows, inp->num_cols);
        Matrix* wa = matrix_dot(&net->weights[i], inp);
        printf("Output shape: (%d, %d)\n", wa->num_rows, wa->num_cols);
        PRINT_MATRIX(wa);

        matrix_free(inp);
        free(inp);

        printf("Matrix add: (%d, %d) x (%d, %d)\n", wa->num_rows, wa->num_cols, *&net->biases[i].num_rows, *&net->biases[i].num_cols);
        Matrix* wab = matrix_add(wa, &net->biases[i]);
        printf("Output shape: (%d, %d)\n", wab->num_rows, wab->num_cols);
        PRINT_MATRIX(wab);

        matrix_free(wa);
        free(wa);

        matrix_sigmoid_(wab);

        inp = wab;
    }

    return inp;
}

// Mini-batch stochastic gradient descent
void stochastic_gradient_descent(int mini_batch_size) {
}
