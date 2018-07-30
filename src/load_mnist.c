#include <stdlib.h>

#include "lib/load_mnist.h"

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
void check_image_header(FILE *image_file) {
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

#ifdef DEBUG
    print_image_header(&header);
#endif
}

#ifdef DEBUG
void print_image_header(MnistImageHeader* header) {
    printf("Image header:\n");
    printf("\tMagic Number: %d\n", header->magic_number);
    printf("\tNumber of images: %d\n", header->num_images);
    printf("\tNumber of rows: %d\n", header->num_rows);
    printf("\tNumber of columns: %d\n", header->num_columns);
}
#endif

/// Read and check the header from an open label file
void check_label_header(FILE *label_file) {
    MnistLabelHeader header;

    fread(&header.magic_number, 4, 1, label_file);
    header.magic_number = reverse_int(header.magic_number);

    fread(&header.num_items, 4, 1, label_file);
    header.num_items = reverse_int(header.num_items);

    if (header.magic_number != 2049) {
        printf("Unexpected value %d as magic number\n", header.magic_number);
        exit(1);
    }

#ifdef DEBUG
    print_label_header(&header);
#endif
}

#ifdef DEBUG
void print_label_header(MnistLabelHeader* header) {
    printf("Label header:\n");
    printf("\tMagic Number: %d\n", header->magic_number);
    printf("\tNumber of Items: %d\n", header->num_items);
}
#endif

/// Read the next image from the images file
MnistImage read_image(FILE* image_file) {
    MnistImage img;

    // Because the struct has only one field, we can read directly into it
    int res = fread(&img, sizeof(MnistImage), 1, image_file);
    if (res != 1) {
        puts("Error reading image from file");
        exit(1);
    }

    return img;
}

/// Read the next label from the labels file
MnistLabel read_label(FILE* label_file) {
    MnistLabel label;

    size_t res = fread(&label, sizeof(MnistLabel), 1, label_file);
    if (res != 1) {
        puts("Failued while reading from label file");
        exit(1);
    }

    return label;
}

/// Opens the file and checks for the expected magic number
/// This advances the read pointer past the header, so as long
/// as the magic number is correct we can immediately start reading
FILE* open_label_file(char* filename) {
    FILE* label_file = fopen(filename, "rb");
    if (label_file == NULL) {
        printf("Failed to open label file %s\n", filename);
        exit(1);
    }

    // Check the label header, advancing the file pointer past the header
    check_label_header(label_file);

    return label_file;
}

FILE* open_image_file(char* filename) {
    FILE* image_file = fopen(filename, "rb");
    if (image_file == NULL) {
        printf("Failed to open label file %s\n", filename);
        exit(1);
    }

    // Check the image header, advancing the file pointer
    check_image_header(image_file);

    return image_file;
}
