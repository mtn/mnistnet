#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "load_mnist.h"
#include "network.h"
#include "macros.h"
#include "util.h"


int main() {
    puts("Hello world!");

    int* sizes = malloc(sizeof(int) * 3);
    sizes[0] = 1;
    sizes[1] = 2;
    sizes[2] = 3;
    Network* net = create_network(3, sizes);
    DEBUG_PRINT(("%d %d %d", net->sizes[0], net->sizes[1], net->sizes[2]));

    free_network(net);

    /* FILE* fp = open_image_file("data/t10k-images-idx3-ubyte"); */

    /* MnistImage img = read_image(fp); */

    /* printf("\n"); */
    /* for (int i = 0; i < 784; i++) { */
    /*     if (i % 28 == 0) { */
    /*         printf("\n"); */
    /*     } */
    /*     if (img.pixels[i] > 0) { */
    /*         printf("1"); */
    /*     } else { */
    /*         printf("%d", img.pixels[i]); */
    /*     } */
    /* } */
    /* printf("\n"); */
}
