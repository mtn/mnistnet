#ifndef __NETWORK_H__
#define __NETWORK_H__

typedef struct {
    // The number of neurons in each layer
    int* sizes;
    int num_layers;

    double** biases;
    double*** weights;
} Network;

Network* create_network(int num_layers, int sizes[]);
void free_network(Network* net);

#endif /* __NETWORK_H__ */
