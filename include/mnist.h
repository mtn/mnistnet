#ifndef __MNIST_H__
#define __MNIST_H__

#include <stdint.h>

#include "util.h"


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
// `end` allows loading less than the full data file into a struct.
// The remaining items get loaded into a second MnistData struct that
// is next to the first in memory.
MnistData* load_data(char* label_filename, char* image_filename, uint32_t end);
void free_mnist_data(MnistData* data);

Matrix* image_to_matrix(MnistImage image);
// Labels are 10-row matrices with value 1 at the label index
Matrix* label_to_matrix(MnistLabel label);

#endif /* __MNIST_H__ */
