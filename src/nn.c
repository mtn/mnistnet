#include <stdio.h>
#include <stdint.h>

#include "lib/load_mnist.h"
#include "lib/macros.h"


int main() {
    puts("Hello world!");

    open_image_file("data/t10k-images-idx3-ubyte");
    open_image_file("data/train-images-idx3-ubyte");
    open_label_file("data/t10k-labels-idx1-ubyte");
    open_label_file("data/train-labels-idx1-ubyte");
}
