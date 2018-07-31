#include <stdlib.h>

#include "lib/network.h"



Network* create_network(int num_layers, int* sizes) {
    Network* net = malloc(sizeof(Network));

    net->num_layers = num_layers;
    net->sizes = sizes;

    net->biases = NULL;
    net->weights = NULL;

    return net;
}

void free_network(Network* net) {
    free(net->sizes);
    free(net->biases);
    free(net->weights);
    free(net);
}


