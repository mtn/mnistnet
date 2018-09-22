#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "network.h"
#include "macros.h"
#include "mnist.h"
#include "nmath.h"
#include "util.h"

#include <assert.h> // TODO removeme

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
    // Each row has a bias matrix (vector)
    net->biases = malloc((net->num_layers - 1) * sizeof(Matrix));
    for (int i = 0; i < net->num_layers - 1; i++) {
        // Initialize a column vector with {# nodes in next row} nodes
        matrix_init(&net->biases[i], net->sizes[i + 1], 1);
        matrix_init_buffer(&net->biases[i], &stdnormal);
    }
}

// Initialize weight vectors
// Assumes that net->sizes and net->num_layers are properly initialized
void init_weights(Network* net) {
    // Each row has a weights matrix
    net->weights = malloc((net->num_layers - 1) * sizeof(Matrix));
    for (int i = 1; i < net->num_layers; i++) {
        matrix_init(&net->weights[i - 1], net->sizes[i], net->sizes[i - 1]);
        /* matrix_init_buffer(&net->weights[i - 1], &stdnormal); */
        matrix_init_buffer(&net->weights[i - 1], &zero);
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

        matrix_free(inp);
        free(inp);

        Matrix* wab = matrix_add(wa, &net->biases[i]);

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

    shuffle_ints_(nums, len);

    return nums;
}

Matrix* init_nabla_b(Network* net) {
    Matrix* nabla_b = malloc((net->num_layers - 1) * sizeof(Matrix));

    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_init_zeros(&nabla_b[i], net->biases[i].num_rows,
                net->biases[i].num_cols);
    }

    return nabla_b;
}

Matrix* init_nabla_w(Network* net) {
    Matrix* nabla_w = malloc((net->num_layers - 1) * sizeof(Matrix));

    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_init_zeros(&nabla_w[i], (&net->weights[i])->num_rows,
                (&net->weights[i])->num_cols);
    }

    return nabla_w;
}

Matrix* cost_derivative(Matrix* output_activations, Matrix* y) {
    return matrix_subtract(output_activations, y);
}

void bp_feed_forward(Network* net, Matrix* activation, Matrix* activations, Matrix* zs) {
    for (int i = 0; i < net->num_layers - 1; i++) {
        Matrix* wa = matrix_dot(&net->weights[i], activation);

        matrix_free(activation);
        free(activation);

        Matrix* z = matrix_add(wa, &net->biases[i]);
        matrix_init_from(&zs[i], z);

        matrix_sigmoid_(z);
        activation = z;

        matrix_init_from(&activations[i + 1], activation);

        matrix_free(wa);
        free(wa);
    }

    matrix_free(activation);
    free(activation);
}

void bp_backwards_pass(Network* net, Matrix* label_vector, Matrix* activations,
        Matrix* zs, Matrix* nabla_b, Matrix* nabla_w) {

    Matrix* cost_der = cost_derivative(&activations[net->num_layers - 1], label_vector);

    Matrix* zs_last = matrix_init_from(NULL, &zs[net->num_layers - 2]);
    matrix_sigmoid_prime_(zs_last);
    Matrix* delta = matrix_hadamard_product(NULL, cost_der, zs_last);

    matrix_init_from(&nabla_b[net->num_layers - 2], delta);

    Matrix* trans = matrix_transpose(&activations[net->num_layers - 2]);
    assert(nabla_w[net->num_layers - 2].elem == NULL);
    matrix_dot_(&nabla_w[net->num_layers - 2], delta, trans);

    matrix_free(zs_last);
    matrix_free(cost_der);
    matrix_free(label_vector);
    matrix_free(trans);
    free(zs_last);
    free(cost_der);
    free(label_vector);
    free(trans);

    for (int i = 2; i < net->num_layers; i++) {
        Matrix* sp = matrix_init_from(NULL, &zs[net->num_layers - 1 - i]);
        matrix_sigmoid_prime_(sp);

        trans = matrix_transpose(&net->weights[net->num_layers - i]);

        double* tmp = delta->elem;
        matrix_dot_(delta, trans, delta);
        free(tmp);

        // Somewhat ugly, but we have to keep a reference to the buffer
        // since the current API doesn't take ownership
        tmp = delta->elem;
        matrix_hadamard_product(delta, delta, sp);
        free(tmp);

        matrix_free(trans);
        matrix_free(sp);
        free(trans);
        free(sp);

        assert(nabla_b[net->num_layers - i - 1].elem == NULL);
        matrix_init_from(&nabla_b[net->num_layers - i - 1], delta);

        trans = matrix_transpose(&activations[net->num_layers - i - 1]);
        Matrix* dotted = matrix_dot(delta, trans);

        assert(nabla_w[net->num_layers - i - 1].elem == NULL);
        matrix_init_from(&nabla_w[net->num_layers - 1 - i], dotted);
        matrix_free(trans);
        matrix_free(dotted);
        free(trans);
        free(dotted);
    }

    matrix_free(delta);
    free(delta);
}

DeltaNabla backprop(Network* net, MnistImage image, MnistLabel label) {
    Matrix* activation = image_to_matrix(image);

    Matrix* nabla_b = calloc(net->num_layers - 1, sizeof(Matrix));
    Matrix* nabla_w = calloc(net->num_layers - 1, sizeof(Matrix));
    Matrix* activations = malloc(net->num_layers * sizeof(Matrix));
    Matrix* zs = malloc((net->num_layers - 1) * sizeof(Matrix));

    Matrix* label_vector = label_to_matrix(label);
    matrix_init_from(&activations[0], activation);

    // Feed forward through the network
    // Takes ownership of activation
    bp_feed_forward(net, activation, activations, zs);

    // Backward pass
    bp_backwards_pass(net, label_vector, activations, zs, nabla_b, nabla_w);

    for (int i = 0; i < net->num_layers; i++) {
        matrix_free(&activations[i]);
    }
    free(activations);

    for (int i = 0; i < net->num_layers - 1; i++) {
        matrix_free(&zs[i]);
    }
    free(zs);

    return (DeltaNabla) { .b = nabla_b, .w = nabla_w };
}

void free_nablas(int num_layers, Matrix* nabla_b, Matrix* nabla_w) {
    for (int i = 0; i < num_layers - 1; i++) {
        matrix_free(&nabla_b[i]);
        matrix_free(&nabla_w[i]);
    }
    free(&nabla_b[0]);
    free(&nabla_w[0]);
}

// note: end is _inclusive_
void update_minibatch(Network* net, MnistData* training_data, Matrix* step,
        int* minibatch_inds, int start, int end) {

    Matrix* nabla_b = init_nabla_b(net);
    Matrix* nabla_w = init_nabla_w(net);

    for (int j = start; j <= end; j++) {
        int ind = minibatch_inds[j];
        DeltaNabla delta = backprop(net, training_data->images[ind],
                training_data->labels[ind]);

        for (int i = 0; i < net->num_layers - 1; i++) {
            matrix_into(&nabla_b[i], matrix_add(&nabla_b[i], &(delta.b)[i]));
            matrix_into(&nabla_w[i], matrix_add(&nabla_w[i], &(delta.w)[i]));

            matrix_free(&(delta.b)[i]);
            matrix_free(&(delta.w)[i]);
        }

        free(delta.b);
        free(delta.w);
    }

    for (int i = 0; i < net->num_layers - 1; i++) {
        Matrix* prod = matrix_hadamard_product(NULL, step, &nabla_w[i]);
        Matrix* sub = matrix_subtract(&net->weights[i], prod);
        matrix_free(prod);
        free(prod);

        // Sub doesn't have to be freed since matrix_into takes ownership
        matrix_into(&net->weights[i], sub);

        prod = matrix_hadamard_product(NULL, step, &nabla_b[i]);
        sub = matrix_subtract(&net->biases[i], prod);
        matrix_free(prod);
        free(prod);

        // sub doesn't need to be freed for the same reason as above
        matrix_into(&net->biases[i], sub);
    }

    free_nablas(net->num_layers, nabla_b, nabla_w);
}

int evaluate(Network* net, MnistData* test_data) {
    int num_correct = 0;
    for (int i = 0; i < test_data->count; i++) {
        Matrix* inp = image_to_matrix(test_data->images[i]);

        // takes ownership of inp
        Matrix* out = feed_forward(net, inp);

        double max = 0;
        for (int j = 0; j < 10; j++) {
            if (max < out->elem[j]) {
                max = out->elem[j];
            }
        }
        int pred = matrix_argmax(out);

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
        int num_epochs, int mini_batch_size, double eta, MnistData* test_data) {

    Matrix* step = matrix_init(NULL, 1, 1);
    step->elem[0] = (double)eta / mini_batch_size;
    int num_batches = training_data->count / mini_batch_size;

    for (int j = 0; j < num_epochs; j++) {
        fprintf(stderr, "Epoch: %d\n", j);

        int* minibatch_inds = get_minibatch_inds(training_data->count);
        for (int i = 0; i < num_batches; i++) {
            int start = mini_batch_size * i;

            if (i % 1000 == 0) {
                fprintf(stderr, "\tUpdating mini batches {%d - %d} / %d\n",
                        i + 1, i + 1000, num_batches);
            }

            update_minibatch(net, training_data, step, minibatch_inds, start,
                    start + mini_batch_size - 1);
        }

        if (test_data != NULL) {
            fprintf(stderr, "Epoch %d: %d / %d\n", j, evaluate(net, test_data),
                    test_data->count);
        } else {
            fprintf(stderr, "Epoch %d complete\n", j);
        }

        free(minibatch_inds);
    }

    matrix_free(step);
    free(step);
}
