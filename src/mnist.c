#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "macros.h"
#include "mnist.h"


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

typedef struct {
    FILE* fp;
    MnistLabelHeader header;
} MnistLabelFile;

typedef struct {
    FILE* fp;
    MnistImageHeader header;
} MnistImageFile;


/// Reverse the byte order of a 32 bit integer
/// This can be done efficiently because we know the size
uint32_t reverse_int(uint32_t i) {
    uint32_t i1 = (i & 0x000000ff) << 24u;
    uint32_t i2 = (i & 0x0000ff00) << 8u;
    uint32_t i3 = (i & 0x00ff0000) >> 8u;
    uint32_t i4 = (i & 0xff000000) >> 24u;

    return (i1 | i2 | i3 | i4);
}

/// Read the header from an open image file
MnistImageHeader check_image_header(FILE *image_file) {
    MnistImageHeader header;

    fread(&header.magic_number, 4, 1, image_file);
    header.magic_number = reverse_int(header.magic_number);

    fread(&header.num_images, 4, 1, image_file);
    header.num_images = reverse_int(header.num_images);

    fread(&header.num_rows, 4, 1, image_file);
    header.num_rows = reverse_int(header.num_rows);

    fread(&header.num_columns, 4, 1, image_file);
    header.num_columns = reverse_int(header.num_columns);

    if (header.magic_number != 2051) {
        printf("Unexpected value %d as magic number\n", header.magic_number);
        exit(1);
    }

    DEBUG_PRINT(("Image header:\n"));
    DEBUG_PRINT(("\tMagic Number: %d\n", header.magic_number));
    DEBUG_PRINT(("\tNumber of images: %d\n", header.num_images));
    DEBUG_PRINT(("\tNumber of rows: %d\n", header.num_rows));
    DEBUG_PRINT(("\tNumber of columns: %d\n", header.num_columns));

    return header;
}

/// Read and check the header from an open label file
MnistLabelHeader check_label_header(FILE *label_file) {
    MnistLabelHeader header;

    fread(&header.magic_number, 4, 1, label_file);
    header.magic_number = reverse_int(header.magic_number);

    fread(&header.num_items, 4, 1, label_file);
    header.num_items = reverse_int(header.num_items);

    if (header.magic_number != 2049) {
        printf("Unexpected value %d as magic number\n", header.magic_number);
        exit(1);
    }

    DEBUG_PRINT(("Label header:\n"));
    DEBUG_PRINT(("\tMagic Number: %d\n", header.magic_number));
    DEBUG_PRINT(("\tNumber of Items: %d\n", header.num_items));

    return header;
}

/// Read the next image from the images file
MnistImage read_image(MnistImageFile* image_file) {
    MnistImage img;

    // Because the struct has only one field, we can read directly into it
    int res = fread(&img, sizeof(MnistImage), 1, image_file->fp);
    if (res != 1) {
        puts("Error reading image from file");
        exit(1);
    }

    return img;
}

/// Read the next label from the labels file
MnistLabel read_label(MnistLabelFile* label_file) {
    MnistLabel label;

    size_t res = fread(&label, sizeof(MnistLabel), 1, label_file->fp);
    if (res != 1) {
        puts("Failed while reading from label file");
        exit(1);
    }

    return label;
}

/// Opens the file and checks for the expected magic number
/// This advances the read pointer past the header, so as long
/// as the magic number is correct we can immediately start reading
MnistLabelFile open_label_file(char* filename) {
    FILE* label_file = fopen(filename, "rb");
    if (label_file == NULL) {
        printf("Failed to open label file %s\n", filename);
        exit(1);
    }

    // Check the label header, advancing the file pointer past the header
    MnistLabelHeader header = check_label_header(label_file);

    return (MnistLabelFile){ .fp = label_file, .header = header };
}

MnistImageFile open_image_file(char* filename) {
    FILE* image_file = fopen(filename, "rb");
    if (image_file == NULL) {
        printf("Failed to open label file %s\n", filename);
        exit(1);
    }

    // Check the image header, advancing the file pointer
    MnistImageHeader header = check_image_header(image_file);

    return (MnistImageFile){ .fp = image_file, .header = header };
}

void init_data(MnistData* data, int count) {
    data->count = count;
    data->images = malloc(count * sizeof(MnistImage));
    data->labels = malloc(count * sizeof(MnistLabel));
}

MnistData* load_data(char* label_filename, char* image_filename, uint32_t end) {
    MnistData* data;

    MnistLabelFile label_file = open_label_file(label_filename);
    MnistImageFile image_file = open_image_file(image_filename);

    assert(label_file.header.num_items == image_file.header.num_images);

    if (end == 0 || end == label_file.header.num_items) {
        end = label_file.header.num_items;
        data = malloc(sizeof(MnistData));
        init_data(data, label_file.header.num_items);
    } else {
        assert(end < label_file.header.num_items);

        // Read the remainder into a second continguous struct
        data = malloc(2 * sizeof(MnistData));
        init_data(&data[0], end);
        init_data(&data[1], label_file.header.num_items - end);
    }

    for (uint32_t i = 0; i < end; i++) {
        data[0].labels[i] = read_label(&label_file);
        data[0].images[i] = read_image(&image_file);
    }
    for (uint32_t i = 0; i < label_file.header.num_items - end; i++) {
        data[1].labels[i] = read_label(&label_file);
        data[1].images[i] = read_image(&image_file);
    }

    return data;
}

void free_mnist_data(MnistData* data) {
    free(data->images);
    free(data->labels);
    // TODO own the data again by changing the api
}

Matrix* image_to_matrix(MnistImage image) {
    Matrix* m = matrix_init(NULL, 784, 1);

    // We have to manually copy because the sizes differ
    for (int i = 0; i < 784; i++) {
        m->elem[i] = (double)image.pixels[i] / 256.0;
    }

    return m;
}

Matrix* label_to_matrix(MnistLabel label) {
    Matrix* m = matrix_init_zeros(NULL, 10, 1);

    m->elem[label] = 1;

    return m;
}
