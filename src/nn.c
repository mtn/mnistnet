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

/*     int* sizes = malloc(sizeof(int) * 3); */
/*     sizes[0] = 1; */
/*     sizes[1] = 2; */
/*     sizes[2] = 3; */
/*     Network* net = create_network(3, sizes); */
/*     DEBUG_PRINT(("%d %d %d", net->sizes[0], net->sizes[1], net->sizes[2])); */

/*     free_network(net); */

    /* Matrix* m1 = malloc(sizeof(Matrix)); */
    /* matrix_init(m1, 2, 3); */
    /* m1->elem[0] = 1; */
    /* m1->elem[1] = 2; */
    /* m1->elem[2] = 3; */
    /* m1->elem[3] = 4; */
    /* m1->elem[4] = 5; */
    /* m1->elem[5] = 6; */

    /* Matrix* m2 = malloc(sizeof(Matrix)); */
    /* matrix_init(m2, 3, 2); */
    /* m2->elem[0] = 7; */
    /* m2->elem[1] = 8; */
    /* m2->elem[2] = 9; */
    /* m2->elem[3] = 10; */
    /* m2->elem[4] = 11; */
    /* m2->elem[5] = 12; */

    /* printf("\n"); */
    /* Matrix* m3 = matrix_multiply(m1, m2); */
    /* for (int i = 0; i < m3->num_rows; i++) { */
    /*     for (int j = 0; j < m3->num_cols; j++) { */
    /*         printf("%f ", m3->elem[matrix_get_ind(m3, i, j)]); */
    /*     } */
    /*     printf("\n"); */
    /* } */
    /* printf("\n"); */

    /* int veclen = 10; */
    /* double* tes9 = malloc(sizeof(double) * veclen); */

    /* for (int i = 0; i < veclen; i++) { */
    /*     test[i] = (double)i; */
    /* } */

    /* sigmoid_(test, veclen); */

    /* printf("\n"); */
    /* for (int i = 0; i < veclen; i++) { */
    /*     printf("%f  ", test[i]); */
    /* } */
    /* printf("\n"); */

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
