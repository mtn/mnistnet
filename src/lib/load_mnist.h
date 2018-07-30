#ifndef __MNIST_H__
#define __MNIST_H__

#include <stdint.h>
#include <stdio.h>

typedef struct {
    uint8_t pixels[784];
} MnistImage;

typedef struct {
    uint32_t magic_number;
    uint32_t num_images;

    uint32_t num_rows;
    uint32_t num_columns;
} MnistImageHeader;

typedef struct {
    uint32_t magic_number;
    uint32_t num_items;
} MnistLabelHeader;

typedef uint8_t MnistLabel;


FILE* open_image_file(char* filename);
FILE* open_label_file(char* filename);

#endif /* __MNIST_H__ */
