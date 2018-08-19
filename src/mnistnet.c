#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "load_mnist.h"
#include "network.h"
#include "macros.h"
#include "nmath.h"
#include "util.h"


int main() {
    puts("Hello world!");

    FILE* image_file = open_image_file("data/train-images-idx3-ubyte");
    
    for (int i = 0; i < 5; i++) {
        MnistImage img = read_image(image_file);

        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                printf("%d", img.pixels[28 * j + k] > 0 ? 1 : 0);
            }
            printf("\n");
        }

        printf("\n\n");
    }
}
