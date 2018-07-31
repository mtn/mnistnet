#include <stdlib.h>
#include <string.h>

#include "network.h"
#include "macros.h"
#include "util.h"

/// Initialize bias vectors
/// Assumes that net->sizes and net->num_layers are properly initialized
void init_biases(Network* net) {
    DEBUG_PRINT(("\nInitializing bias vectors:\n"));
    net->biases = malloc((net->num_layers - 1) * sizeof(int*));
    for (int i = 0; i < net->num_layers - 1; i++) {
        net->biases[i] = malloc(net->sizes[i + 1] * sizeof(int));
        doublearr_init(net->sizes[i + 1], net->biases[i], &stdnormal);

        DEBUG_PRINT(("<"));
        for (int j = 0; j < net->sizes[i + 1] - 1; j++) {
            DEBUG_PRINT(("%f, ", net->biases[i][j]));
        }
        DEBUG_PRINT(("%f>\n", net->biases[i][net->sizes[ i+ 1] - 1]));
    }
}

/// Initialize weight vectors
/// Assumes that net->sizes and net->num_layers are properly initialized
void init_weights(Network* net) {
    DEBUG_PRINT(("\nInitializing weights:\n"));
    // Index into a row first
    net->weights = malloc((net->num_layers - 1) * sizeof(int**));
    for (int i = 1; i < net->num_layers; i++) {
        DEBUG_PRINT(("Layer %d-%d:\n <\n", i, i + 1));
        // Then index into a node
        net->weights[i] = malloc(net->sizes[i] * sizeof(int));
        for (int j = 0; j < net->sizes[i]; j++) {
            // Then index into a weight of that node (going backwards)
            net->weights[i][j] = malloc(net->sizes[i - 1] * sizeof(int));
            doublearr_init(net->sizes[i - 1], net->weights[i][j], &stdnormal);

            DEBUG_PRINT(("\t <"));
            for (int k = 0; k < net->sizes[i - 1] - 1; k++) {
                DEBUG_PRINT(("%f, ", net->weights[i][j][k]));
            }
            DEBUG_PRINT(("%f>\n ", net->weights[i][j][net->sizes[i - 1] - 1]));
        }
        DEBUG_PRINT((">\n"));
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

    for (int i = 0; i < net->num_layers - 1; i++) {
        free(net->biases[i]);
    }
    free(net->biases);

    free(net->weights);
    free(net);
}
