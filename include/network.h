#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "mnist.h"
#include "util.h"


typedef struct {
    // The number of neurons in each layer
    int* sizes;
    int num_layers;

    // Biases are a set of 1 dimensional matrices (vectors)
    Matrix* biases;
    Matrix* weights;
} Network;

Network* create_network(int num_layers, int sizes[]);
void free_network(Network* net);

void stochastic_gradient_descent(Network* net, MnistData* training_data,
        int num_epochs, int mini_batch_size, double eta, MnistData* test_data);

Matrix* feed_forward(Network* net, Matrix* inp);


#endif /* __NETWORK_H__ */
