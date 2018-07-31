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


/// Open the files and check that the headers are correct.
/// Filename is relative to the root (where the binary goes).
/// If these succeed, the data should formatted correctly.
FILE* open_image_file(char* filename);
FILE* open_label_file(char* filename);

/// Grab the next image or label from an opened file.
/// These should be initially called on file descriptors
/// returned from the above two functions.
MnistImage read_image(FILE* image_file);
MnistLabel read_label(FILE* label_file);


#endif /* __MNIST_H__ */
