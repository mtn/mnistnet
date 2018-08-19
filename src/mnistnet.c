#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "mnist.h"
#include "network.h"
#include "macros.h"
#include "nmath.h"
#include "util.h"


int main() {
    puts("Hello world!");

    MnistData* training_data = load_data("data/train-labels-idx1-ubyte",
                                         "data/train-images-idx3-ubyte");

    PRINT_DATAHEAD((training_data));
}
