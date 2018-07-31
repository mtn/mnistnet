#ifndef __NETWORK_H__
#define __NETWORK_H__

typedef struct {
    // The number of neurons in each layer
    int* sizes;
    int num_layers;

    int* biases;
    int* weights;
} Network;

Network* create_network();

#endif /* __NETWORK_H__ */
