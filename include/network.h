#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "util.h"


typedef struct {
    // The number of neurons in each layer
    int* sizes;
    int num_layers;

    // Biases are a set of 1 dimensional matrices (vectors)
    Matrix* biases;
    Matrix** weights;
} Network;

Network* create_network(int num_layers, int sizes[]);
void free_network(Network* net);

Matrix* feed_forward(Network* net, Matrix* inp);


#endif /* __NETWORK_H__ */
