#ifndef __MNIST_H__
#define __MNIST_H__

#include <stdint.h>
#include <stdio.h>


typedef struct {
    uint8_t pixels[784];
} MnistImage;

typedef uint8_t MnistLabel;

// Loaded data, suitable for handling by models
typedef struct {
    MnistImage* images;
    MnistLabel* labels;

    // Number of labels and images should be the same
    int count;
} MnistData;


// Load data in memory from file names
MnistData* load_data(char* label_filename, char* image_filename);
void free_mnist_data(MnistData* data);

#endif /* __MNIST_H__ */
