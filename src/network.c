#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "network.h"
#include "macros.h"
#include "mnist.h"
#include "nmath.h"
#include "util.h"


// No need to keep track of the length, sine we know it from the net
typedef struct {
    Matrix* delta_b;
    Matrix* delta_w;
} DeltaNabla;


double one_init() {
    return 1;
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
        /* matrix_init_buffer(&net->biases[i], &stdnormal); */
        matrix_init_buffer(&net->biases[i], &one_init); // TODO revert

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
        /* matrix_init_buffer(&net->weights[i - 1], &stdnormal); */
        matrix_init_buffer(&net->weights[i - 1], &one_init); // TODO revert

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

#include <assert.h>
DeltaNabla backprop(Network* net, MnistImage image, MnistLabel label) {
    // TODO these buffers are leaking
    Matrix* nabla_b = init_nabla_b(net);
    /* for (int i = 0; i < net->num_layers - 1; i++) { */
    /*     printf("%d x %d\n", (&nabla_b[i])->num_rows, (&nabla_b[i])->num_cols); */
    /* } */
    Matrix* nabla_w = init_nabla_w(net);
    assert((&nabla_w[0])->num_rows == 30);
    assert((&nabla_w[0])->num_cols == 784);
    assert((&nabla_w[1])->num_rows == 10);
    assert((&nabla_w[1])->num_cols == 30);

    Matrix* activation = image_to_matrix(image);
    Matrix* activations = malloc(net->num_layers * sizeof(Matrix));
    matrix_init_from(&activations[0], activation);

    Matrix* zs = malloc((net->num_layers - 1) * sizeof(Matrix));

    // Feed forward
    for (int i = 0; i < net->num_layers - 1; i++) {
        Matrix* wa = matrix_dot(&net->weights[i], activation);
        DEBUG_PRINT(("w * a: \n"));
        PRINT_MATRIX(wa);

        matrix_free(activation);
        free(activation);

        Matrix* z = matrix_add(wa, &net->biases[i]);
        matrix_init_from(&zs[i], z);

        matrix_sigmoid_(z);
        activation = z;
        matrix_init_from(&activations[i + 1], activation);

        PRINT_MATRIX((&activations[i + 1]));
    }

    // Backward pass
    Matrix* label_vector = label_to_matrix(label);

    Matrix* cost_der = cost_derivative(&activations[net->num_layers - 1], label_vector);

    Matrix* zs_last = matrix_init_from(NULL, &zs[net->num_layers - 2]);
    matrix_sigmoid_prime_(zs_last);

    Matrix* delta = matrix_hadamard_product(NULL, cost_der, zs_last);

    matrix_init_from(&nabla_b[net->num_layers - 2], delta);

    Matrix* trans = matrix_transpose(&activations[net->num_layers - 2], false);
    matrix_dot_(&nabla_w[net->num_layers - 2], delta, trans);

    for (int i = 2; i < net->num_layers; i++) {
        Matrix* sp = matrix_init_from(NULL, &zs[net->num_layers - i - 1]);
        assert((&zs[0])->num_rows == 30);
        assert((&zs[1])->num_rows == 10);
        matrix_sigmoid_prime_(sp);
        assert(sp->num_rows == 30);

        trans = matrix_transpose(&net->weights[net->num_layers - i], false);
        matrix_dot_(delta, trans, delta);
        matrix_hadamard_product(delta, delta, sp);
        assert(delta->num_rows == 30);

        assert((&nabla_b[1])->num_rows == 10);
        matrix_init_from(&nabla_b[net->num_layers - i - 1], delta);
        assert((&nabla_b[1])->num_rows == 10);

        matrix_free(trans);
        free(trans);

        assert((&activations[net->num_layers - i - 1])->num_rows == 784);
        assert((&activations[net->num_layers - i - 1])->num_cols == 1);
        trans = matrix_transpose(&activations[net->num_layers - i - 1], false);
        assert(trans->num_rows == 1);
        assert(trans->num_cols == 784);

        Matrix* dotted = matrix_dot(delta, trans);
        assert(dotted->num_rows == 30);
        assert(dotted->num_cols == 784);
        matrix_init_from(&nabla_w[net->num_layers - i - 1], dotted);
        assert((&nabla_w[0])->num_rows == 30);
        assert((&nabla_w[0])->num_cols == 784);
        /* printf("%d\n", (&nabla_w[1])->num_rows); */
        /* printf("%d\n", (&nabla_w[1])->num_cols); */
        assert((&nabla_w[1])->num_rows == 10);
        assert((&nabla_w[1])->num_cols == 30);

        matrix_free(dotted);
        free(dotted);

        matrix_free(trans);
        free(trans);
    }

    matrix_free(zs_last);
    free(zs_last);

    matrix_free(cost_der);
    free(cost_der);

    matrix_free(label_vector);
    free(label_vector);

    for (int i = 0; i < net->num_layers; i++) {
        matrix_free(&activations[i]);
    }
    free(activations);

    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_free(&zs[i]);
    }
    free(zs);

    /* for (int i = 0; i < net->num_layers - 1; i++) { */
    /*     printf("%d x %d\n", (&nabla_b[i])->num_rows, (&nabla_b[i])->num_cols); */
    /* } */
    return (DeltaNabla) { .delta_b = nabla_b, .delta_w = nabla_w };
}

// note: end is _exclusive_
void update_minibatch(Network* net, MnistData* training_data, int eta,
        int* minibatch_inds, int start, int end) {

    Matrix* nabla_b = init_nabla_b(net);
    Matrix* nabla_w = init_nabla_w(net);

    Matrix* step = matrix_init(NULL, 1, 1);
    step->elem[0] = eta / (end - start);

    for (int i = start; i <= end; i++) {
        int ind = minibatch_inds[i];
        /* printf("starting backprop %d\n", i); */
        DeltaNabla delta = backprop(net, training_data->images[ind],
                training_data->labels[ind]);
        /* puts("made it out of backprop"); */

        for (int i = 0; i < net->num_layers - 1; i++) {
            matrix_into(&nabla_b[i], matrix_add(&nabla_b[i], &(delta.delta_b)[i]));
            matrix_into(&nabla_w[i], matrix_add(&nabla_w[i], &(delta.delta_w)[i]));
        }

        for (int i = 0; i < net->num_layers - 1; i++) {
            Matrix* sub = matrix_subtract(&net->weights[i], step);
            matrix_hadamard_product(sub, sub, &nabla_w[i]);
            matrix_into(&net->weights[i], sub);

            sub = matrix_subtract(&net->biases[i], step);
            matrix_hadamard_product(sub, sub, &nabla_b[i]);
            matrix_into(&net->biases[i], sub);
        }
    }

    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_free(&nabla_b[i]);
        matrix_free(&nabla_w[i]);
    }
}

// TODO add evaluation
int evaluate(Network* net, MnistData* test_data) {
    return 0;
}

// Mini-batch stochastic gradient descent
// test_data can be NULL (in which case the network isn't evaluated after each epoch)
void stochastic_gradient_descent(Network* net, MnistData* training_data,
        int num_epochs, int mini_batch_size, int eta, MnistData* test_data) {

    puts("Starting SGD");
    for (int j = 0; j < num_epochs; j++) {
        printf("Epoch %d\n", j);
        puts("Getting minibatch inds");
        int* minibatch_inds = get_minibatch_inds(training_data->count);
        puts("Done getting minibatch inds");
        int num_batches = training_data->count / mini_batch_size;
        puts("Computed number of batches");

        for (int i = 0; i < num_batches; i++) {
            int start = mini_batch_size * i;
            /* puts("Updating minibatch"); */
            update_minibatch(net, training_data, eta, minibatch_inds, start,
                    start + mini_batch_size - 1);
            /* puts("Finished updating minibatch"); */
        }

        if (test_data != NULL) {
            printf("Epoch %d: %d / %d", j, evaluate(net, test_data), test_data->count);
        } else {
            printf("Epoch %d complete\n", j);
        }
    }
}
