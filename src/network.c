#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "network.h"
#include "macros.h"
#include "mnist.h"
#include "nmath.h"
#include "util.h"


#include <assert.h> // TODO

// No need to keep track of the length, sine we know it from the net
typedef struct {
    Matrix* b;
    Matrix* w;
} DeltaNabla;


double zero() {
    return 0;
}

// Initialize bias vectors
// Assumes that net->sizes and net->num_layers are properly initialized
void init_biases(Network* net) {
    DEBUG_PRINT(("\nInitializing bias vectors:\n"));

    // Each row has a bias matrix (vector)
    net->biases = malloc((net->num_layers - 1) * sizeof(Matrix));
    for (int i = 0; i < net->num_layers - 1; i++) {
        // Initialize a column vector with {# nodes in next row} nodes
        matrix_init(&net->biases[i], net->sizes[i + 1], 1);
        matrix_init_buffer(&net->biases[i], &stdnormal);
        /* matrix_init_buffer(&net->biases[i], &zero); */

        PRINT_MATRIX((&net->biases[i]));
    }
}

// Initialize weight vectors
// Assumes that net->sizes and net->num_layers are properly initialized
void init_weights(Network* net) {
    DEBUG_PRINT(("\nInitializing weights:\n"));

    // Each row has a weights matrix
    net->weights = malloc((net->num_layers - 1) * sizeof(Matrix));
    for (int i = 1; i < net->num_layers; i++) {
        DEBUG_PRINT(("Layer %d-%d:\n", i, i + 1));

        matrix_init(&net->weights[i - 1], net->sizes[i], net->sizes[i - 1]);
        matrix_init_buffer(&net->weights[i - 1], &stdnormal);
        /* matrix_init_buffer(&net->weights[i - 1], &zero); */

        PRINT_MATRIX((&net->weights[i - 1]));
    }
}

Network* create_network(int num_layers, int sizes[]) {
    Network* net = malloc(sizeof(Network));

    net->num_layers = num_layers;

    net->sizes = sizes;

    init_biases(net);
    init_weights(net);

    return net;
}

void free_network(Network* net) {
    // Free the memory associated with each bias matrix
    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_free(&net->biases[i]);
    }
    free(net->biases);

    // Free the memory associated with each weight matrix
    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_free(&net->weights[i]);
    }
    // Free the memory associated with each weight matrix
    free(net->weights);

    free(net);
}

// Feedforward
Matrix* feed_forward(Network* net, Matrix* inp) {
    for (int i = 0; i < net->num_layers - 1; i++) {
        Matrix* wa = matrix_dot(&net->weights[i], inp);
        DEBUG_PRINT(("w * a: \n"));
        PRINT_MATRIX(wa);

        matrix_free(inp);
        free(inp);

        Matrix* wab = matrix_add(wa, &net->biases[i]);
        DEBUG_PRINT(("wa + b\n"));
        PRINT_MATRIX(wab);

        matrix_free(wa);
        free(wa);

        matrix_sigmoid_(wab);

        inp = wab;
    }

    return inp;
}

int* get_minibatch_inds(int len) {
    int* nums = malloc(len * sizeof(int));

    for (int i = 0; i < len; i++) {
        nums[i] = i;
    }

    puts("Shuffling ints");
    shuffle_ints_(nums, len);
    puts("Done shuffling ints");

    return nums;
}

Matrix* init_nabla_b(Network* net) {
    Matrix* nabla_b = malloc(net->num_layers * sizeof(Matrix));

    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_init_zeros(&nabla_b[i], net->biases[i].num_rows,
                net->biases[i].num_cols);

        PRINT_MATRIX((&nabla_b[i]));
    }

    return nabla_b;
}

Matrix* init_nabla_w(Network* net) {
    Matrix* nabla_w = malloc(net->num_layers * sizeof(Matrix));

    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_init_zeros(&nabla_w[i], (&net->weights[i])->num_rows,
                (&net->weights[i])->num_cols);

        PRINT_MATRIX((&nabla_w[i]));
    }

    return nabla_w;
}

Matrix* cost_derivative(Matrix* output_activations, Matrix* y) {
    return matrix_subtract(output_activations, y);
}

DeltaNabla backprop(Network* net, MnistImage image, MnistLabel label) {
    // TODO these buffers might be leaking
    Matrix* nabla_b = init_nabla_b(net);
    Matrix* nabla_w = init_nabla_w(net);
    Matrix* label_vector = label_to_matrix(label);

    Matrix* activation = image_to_matrix(image);

    Matrix* activations = malloc(net->num_layers * sizeof(Matrix));
    matrix_into(&activations[0], activation);

    Matrix* zs = malloc((net->num_layers - 1) * sizeof(Matrix));

    // Feed forward
    for (int i = 0; i < net->num_layers - 1; i++) {
        Matrix* wa = matrix_dot(&net->weights[i], activation);

        DEBUG_PRINT(("w * a: \n"));
        PRINT_MATRIX(wa);

        Matrix* z = matrix_add(wa, &net->biases[i]);
        matrix_init_from(&zs[i], z);

        matrix_sigmoid_(z);
        activation = z;
        matrix_into(&activations[i + 1], activation);

        PRINT_MATRIX((&activations[i + 1]));
    }

    /* for (int i = 0; i < net->num_layers; i++) { */
    /*     double max; */
    /*     for (int q = 0; q < (&activations[i])->num_rows * (&activations[i])->num_cols; q++) { */
    /*         if ((&activations[i])->elem[q] > max) { */
    /*             max = (&activations[i])->elem[q]; */
    /*         } */
    /*     } */

    /*     assert(max > 0.000001); */
    /*     /1* printf("MAX %f\n", max); *1/ */
    /* } */

    // Backward pass
    Matrix* cost_der = cost_derivative(&activations[net->num_layers - 1], label_vector);

    /* max = 0; */
    /* for (int q = 0; q < (cost_der)->num_rows * (cost_der)->num_cols; q++) { */
    /*     if (cost_der->elem[q] > max) { */
    /*         max = (cost_der)->elem[q]; */
    /*     } */
    /* } */

    /* printf("MAX %f\n", max); */
    /* P_MATRIX((&activations[net->num_layers - 1])); */
    /* P_MATRIX(label_vector); */
    /* P_MATRIX(cost_der); */
    /* assert(max != 0); */

    Matrix* zs_last = matrix_init_from(NULL, &zs[net->num_layers - 2]);
    matrix_sigmoid_prime_(zs_last);
    Matrix* delta = matrix_hadamard_product(NULL, cost_der, zs_last);

    matrix_init_from(&nabla_b[net->num_layers - 2], delta);

    Matrix* trans = matrix_transpose(&activations[net->num_layers - 2]);
    matrix_dot_(&nabla_w[net->num_layers - 2], delta, trans);

    matrix_free(zs_last);
    matrix_free(cost_der);
    matrix_free(label_vector);
    free(zs_last);
    free(cost_der);
    free(label_vector);

    for (int i = 2; i < net->num_layers; i++) {
        Matrix* sp = matrix_init_from(NULL, &zs[net->num_layers - 1 - i]);
        matrix_sigmoid_prime_(sp);

        trans = matrix_transpose(&net->weights[net->num_layers - i]);
        matrix_dot_(delta, trans, delta);
        matrix_hadamard_product(delta, delta, sp);

        matrix_free(trans);
        free(trans);

        matrix_init_from(&nabla_b[net->num_layers - i - 1], delta);

        // TODO into rather than deallocating matrix?
        trans = matrix_transpose(&activations[net->num_layers - i - 1]);
        Matrix* dotted = matrix_dot(delta, trans);

        matrix_init_from(&nabla_w[net->num_layers - 1 - i], dotted);

        matrix_free(trans);
        matrix_free(dotted);
        free(trans);
        free(dotted);
    }

    for (int i = 0; i < net->num_layers; i++) {
        matrix_free(&activations[i]);
    }
    free(activations);

    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_free(&zs[i]);
    }
    free(zs);

/*     for (int i = 0; i < net->num_layers - 1; i++) { */
/*         double min = 0; */
/*         for (int q = 0; q < (&nabla_b[i])->num_rows * (&nabla_b[i])->num_cols; q++) { */
/*             if ((&nabla_b[i])->elem[q] < min) { */
/*                 min = (&nabla_b[i])->elem[q]; */
/*             } */
/*         } */

/*         assert(min <  0); */
/*     } */

/*     for (int i = 0; i < net->num_layers - 1; i++) { */
/*         double min= 0; */
/*         for (int q = 0; q < (&nabla_w[i])->num_rows * (&nabla_w[i])->num_cols; q++) { */
/*             if ((&nabla_w[i])->elem[q] < min) { */
/*                 min = (&nabla_w[i])->elem[q]; */
/*             } */
/*         } */

/*         assert(min<0); */
/*     } */

    return (DeltaNabla) { .b = nabla_b, .w = nabla_w };
}

// note: end is _inclusive_
void update_minibatch(Network* net, MnistData* training_data, int eta,
        int* minibatch_inds, int start, int end) {

    /* printf("Updating minibatch, inds %d to %d\n", start, end); */

    Matrix* nabla_b = init_nabla_b(net);
    Matrix* nabla_w = init_nabla_w(net);

    // TODO allocate this outside, rather than passing arg eta
    Matrix* step = matrix_init(NULL, 1, 1);
    step->elem[0] = (double)eta / (end - start);

    for (int j = start; j <= end; j++) {
        int ind = minibatch_inds[j];
        DeltaNabla delta = backprop(net, training_data->images[ind],
                training_data->labels[ind]);

        /* exit(1); */

        /* Matrix* nabla_b = delta.b; */
        /* Matrix* nabla_w = delta.w; */
        /* for (int i = 0; i < net->num_layers - 1; i++) { */
        /*     double max; */
        /*     for (int q = 0; q < (&nabla_w[i])->num_rows * (&nabla_w[i])->num_cols; q++) { */
        /*         if ((&nabla_w[i])->elem[q] > max) { */
        /*             max = (&nabla_w[i])->elem[q]; */
        /*         } */
        /*     } */

        /*     assert(max > 0.000001); */
        /*     printf("MAX %f\n", max); */
        /* } */

        // TODO possible memory leak
        for (int i = 0; i < net->num_layers - 1; i++) {
            matrix_into(&nabla_b[i], matrix_add(&nabla_b[i], &(delta.b)[i]));
            matrix_into(&nabla_w[i], matrix_add(&nabla_w[i], &(delta.w)[i]));
        }

        for (int i = 0; i < net->num_layers - 1; i++) {
            Matrix* prod = matrix_hadamard_product(NULL, step, &nabla_w[i]);
            Matrix* sub = matrix_subtract(&net->weights[i], prod);
            matrix_into(&net->weights[i], sub);

            matrix_free(prod);
            free(prod);

            prod = matrix_hadamard_product(NULL, step, &nabla_b[i]);
            sub = matrix_subtract(&net->biases[i], prod);
            matrix_into(&net->biases[i], sub);

            matrix_free(prod);
            free(prod);
        }
    }

    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_free(&nabla_b[i]);
        matrix_free(&nabla_w[i]);
    }
    free(&nabla_b[0]);
    free(&nabla_w[0]);

    matrix_free(step);
    free(step);
}

int evaluate(Network* net, MnistData* test_data) {
    int num_correct = 0;
    for (int i = 0; i < test_data->count; i++) {
        Matrix* inp = image_to_matrix(test_data->images[i]);
        /* LOL_MNIST_IMG(test_data->images[i]); */
        // takes ownership of inp
        Matrix* out = feed_forward(net, inp);

        double max = 0;
        for (int j = 0; j < 10; j++) {
            if (max < out->elem[j]) {
                max = out->elem[j];
            }
        }
        assert(max > 0);
        /* PRINT_MATRIX(out); */
        /* printf("\n"); */
        int pred = matrix_argmax(out);

        /* P_MATRIX(out); */

        printf("Predicted %d: %f %f actual %d\n", pred, out->elem[pred], max, test_data->labels[i]);
        if (pred == test_data->labels[i]) {
            num_correct++;
        }

        matrix_free(out);
        free(out);
    }

    return num_correct;
}

// Mini-batch stochastic gradient descent
// test_data can be NULL (in which case the network isn't evaluated after each epoch)
void stochastic_gradient_descent(Network* net, MnistData* training_data,
        int num_epochs, int mini_batch_size, int eta, MnistData* test_data) {

    puts("Starting SGD");
    for (int j = 0; j < num_epochs; j++) {
        printf("Epoch %d\n", j + 1);
        int* minibatch_inds = get_minibatch_inds(training_data->count);

        /* for (int i = 0; i < training_data->count; i++) { */
        /*     assert(minibatch_inds[i] == i); */
        /* } */

        int num_batches = training_data->count / mini_batch_size;

        for (int i = 0; i < num_batches; i++) {
            if (i % 100 == 0) {
                printf("Epoch %d Updating minibatch %d\n", j + 1, i);
            }

            int start = mini_batch_size * i;
            update_minibatch(net, training_data, eta, minibatch_inds, start,
                    start + mini_batch_size - 1);
            /* exit(1); */
        }

        if (test_data != NULL) {
            printf("Epoch %d: %d / %d\n", j + 1, evaluate(net, test_data), test_data->count);
        } else {
            printf("Epoch %d complete\n", j + 1);
        }
    }
}
