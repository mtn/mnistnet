#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <errno.h>

#include "network.h"
#include "macros.h"
#include "mnist.h"
#include "nmath.h"
#include "util.h"

#ifndef TESTVERBSOE
#define TESTVERBOSE false
#endif


void test_mnist_loader() {
    MnistData* training_data = load_data("data/train-labels-idx1-ubyte",
                                         "data/train-images-idx3-ubyte");

    MnistData* test_data = load_data("data/t10k-labels-idx1-ubyte",
                                     "data/t10k-images-idx3-ubyte");

    // No assertions, just check that this runs
    // Magic numbers are checked normally at runtime
    // Output will be suppressed, except in verbose mode
    PRINT_DATAHEAD((training_data));
    PRINT_DATAHEAD((test_data));

    free_mnist_data(training_data);
    free_mnist_data(test_data);
}

void test_network_init() {
    int* sizes = malloc(sizeof(int) * 3);

    sizes[0] = 1;
    sizes[1] = 2;
    sizes[2] = 3;

    Network* net = create_network(3, sizes);

    assert(net->sizes[0] == 1);
    assert(net->sizes[1] == 2);
    assert(net->sizes[2] == 3);

    free_network(net);
}

void test_matrix_init_from() {
    DEBUG_PRINT(("From matrix:\n"));
    Matrix* from = matrix_init(NULL, 2, 2);
    for (int i = 0; i < 4; i++) {
        from->elem[i] = (double)i;
    }

    PRINT_MATRIX(from);

    Matrix* new = matrix_init_from(NULL, from);

    DEBUG_PRINT(("New matrix:\n"));
    PRINT_MATRIX(new);

    assert(new->num_rows == 2);
    assert(new->num_cols == 2);
    for (int i = 0; i < 4; i++) {
        assert(new->elem[i] - (double)i <= 0.01);
    }

    matrix_free(from);
    matrix_free(new);
    free(from);
    free(new);
}

void test_matrix_inner_product() {
    DEBUG_PRINT(("Computing matrix inner product:\n"));

    Matrix* m1 = matrix_init(NULL, 1, 3);
    m1->elem[0] = 1;
    m1->elem[1] = 1;
    m1->elem[2] = 1;

    Matrix* m2 = matrix_init(NULL, 3, 1);
    m2->elem[0] = 1;
    m2->elem[1] = 1;
    m2->elem[2] = 1;

    PRINT_MATRIX(m1);
    DEBUG_PRINT(("x\n"));
    PRINT_MATRIX(m2);

    Matrix* m3 = matrix_dot(m1, m2);

    DEBUG_PRINT(("Result:\n"));
    PRINT_MATRIX(m3);

    assert(m3->num_cols == 1);
    assert(m3->num_rows == 1);
    assert(m3->elem[0] == 3);

    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    free(m1);
    free(m2);
    free(m3);
}

void test_matrix_init_zeros() {
    Matrix* m = matrix_init_zeros(NULL, 2, 2);

    assert(m->num_rows == 2);
    assert(m->num_cols == 2);

    for (int i = 0; i < 4; i++) {
        assert(m->elem[i] == 0);
    }

    matrix_free(m);
    free(m);
}

void test_matrix_inner_product_fail() {
    Matrix* m1 = matrix_init(NULL, 1, 3);
    m1->elem[0] = 1;
    m1->elem[1] = 1;
    m1->elem[2] = 1;

    Matrix* m2 = matrix_init(NULL, 1, 3);
    m2->elem[0] = 1;
    m2->elem[1] = 1;
    m2->elem[2] = 1;

    // Expected failure with exit status 1
    matrix_dot(m1, m2);
}

void test_matrix_dot() {
    Matrix* m1 = matrix_init(NULL, 2, 3);
    m1->elem[0] = 1;
    m1->elem[1] = 2;
    m1->elem[2] = 3;
    m1->elem[3] = 4;
    m1->elem[4] = 5;
    m1->elem[5] = 6;

    assert(m1 != NULL);
    assert(m1->num_rows == 2);
    assert(m1->num_cols == 3);

    Matrix* m2 = matrix_init(NULL, 3, 2);
    m2->elem[0] = 7;
    m2->elem[1] = 8;
    m2->elem[2] = 9;
    m2->elem[3] = 10;
    m2->elem[4] = 11;
    m2->elem[5] = 12;

    assert(m2 != NULL);
    assert(m2->num_rows == 3);
    assert(m2->num_cols == 2);

    Matrix* m3 = matrix_dot(m1, m2);

    assert(m3->num_rows == 2);
    assert(m3->num_cols == 2);
    assert(m3->elem[0] == 58);
    assert(m3->elem[1] == 64);
    assert(m3->elem[2] == 139);
    assert(m3->elem[3] == 154);

    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    free(m1);
    free(m2);
    free(m3);
}

void test_matrix_dot_2() {
    Matrix* a = matrix_init(NULL, 2, 2);
    for (int i = 0; i < 4; i++) {
        a->elem[i] = (double) i + 1;
    }

    Matrix* b = matrix_init(NULL, 2, 2);
    for (int i = 0; i < 4; i++) {
        b->elem[i] = (double) i + 1;
    }

    DEBUG_PRINT(("Computing matrix multiplication:\n"));
    PRINT_MATRIX(a);
    DEBUG_PRINT(("x\n"));
    PRINT_MATRIX(b);

    Matrix* c = matrix_dot(a, b);

    DEBUG_PRINT(("Result: \n"));
    PRINT_MATRIX(c);

    assert(c->num_rows == 2);
    assert(c->num_cols == 2);

    assert(c->elem[0] == 7);
    assert(c->elem[1] == 10);
    assert(c->elem[2] == 15);
    assert(c->elem[3] == 22);
}

void test_matrix_dot_under_aliasing() {
    Matrix* a = matrix_init(NULL, 2, 2);
    for (int i = 0; i < 4; i++) {
        a->elem[i] = (double) i + 1;
    }

    a = matrix_dot_(a, a, a);

    PRINT_MATRIX(a);

    assert(a->elem[0] == 7);
    assert(a->elem[1] == 10);
    assert(a->elem[2] == 15);
    assert(a->elem[3] == 22);
}

void test_matrix_times_scalar_right() {
    Matrix* m1 = matrix_init(NULL, 2, 3);
    m1->elem[0] = 1;
    m1->elem[1] = 2;
    m1->elem[2] = 3;
    m1->elem[3] = 4;
    m1->elem[4] = 5;
    m1->elem[5] = 6;

    assert(m1 != NULL);
    assert(m1->num_rows == 2);
    assert(m1->num_cols == 3);

    Matrix* m2 = matrix_init(NULL, 1, 1);
    m2->elem[0] = 2;

    Matrix* m3 = matrix_dot(m1, m2);
    assert(m3 != NULL);
    assert(m3->num_rows == 2);
    assert(m3->num_cols == 3);

    assert(m3->elem[0] == 2);
    assert(m3->elem[1] == 4);
    assert(m3->elem[2] == 6);
    assert(m3->elem[3] == 8);
    assert(m3->elem[4] == 10);
    assert(m3->elem[5] == 12);

    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    free(m1);
    free(m2);
    free(m3);
}

void test_matrix_times_scalar_left() {
    Matrix* m2 = matrix_init(NULL, 2, 3);
    m2->elem[0] = 1;
    m2->elem[1] = 2;
    m2->elem[2] = 3;
    m2->elem[3] = 4;
    m2->elem[4] = 5;
    m2->elem[5] = 6;

    assert(m2 != NULL);
    assert(m2->num_rows == 2);
    assert(m2->num_cols == 3);

    Matrix* m1 = matrix_init(NULL, 1, 1);
    m1->elem[0] = 2;

    Matrix* m3 = matrix_dot(m1, m2);
    assert(m3 != NULL);
    assert(m3->num_rows == 2);
    assert(m3->num_cols == 3);
    assert(m3->elem[0] == 2);
    assert(m3->elem[1] == 4);
    assert(m3->elem[2] == 6);
    assert(m3->elem[3] == 8);
    assert(m3->elem[4] == 10);
    assert(m3->elem[5] == 12);

    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    free(m1);
    free(m2);
    free(m3);
}

void test_matrix_hadamard_product() {
    Matrix* m = matrix_init(NULL, 4, 1);
    for (int i = 0; i < 4; i++) {
        m->elem[i] = (double) i + 1;
    }

    Matrix* m1 = matrix_hadamard_product(NULL, m, m);

    assert(m1->num_rows == 4);
    assert(m1->num_cols == 1);

    assert(m1->elem[0] == 1);
    assert(m1->elem[1] == 4);
    assert(m1->elem[2] == 9);
    assert(m1->elem[3] == 16);

    matrix_free(m);
    free(m);
    matrix_free(m1);
    free(m1);
}

void test_matrix_hadamard_product_broadcasting() {
    Matrix* m = matrix_init(NULL, 4, 1);
    for (int i = 0; i < 4; i++) {
        m->elem[i] = (double) i + 1;
    }

    Matrix* m1 = matrix_init(NULL, 1, 1);
    m1->elem[0] = (double) 2;

    Matrix* m2 = matrix_hadamard_product(NULL, m, m1);

    assert(m2->num_rows == 4);
    assert(m2->num_cols == 1);

    assert(m2->elem[0] == 2);
    assert(m2->elem[1] == 4);
    assert(m2->elem[2] == 6);
    assert(m2->elem[3] == 8);

    matrix_free(m);
    free(m);
    matrix_free(m1);
    free(m1);
    matrix_free(m2);
    free(m2);
}

void test_matrix_add_same_dimensions() {
    Matrix* m = matrix_init(NULL, 2, 2);
    m->elem[0] = 1;
    m->elem[1] = 2;
    m->elem[2] = 3;
    m->elem[3] = 4;

    // There shouldn't be problems with memory due to passing the same matrix
    Matrix* m1 = matrix_add(m, m);
    assert(m1 != NULL);
    assert(m1->num_rows == 2);
    assert(m1->num_cols == 2);
    assert(m1->elem[0] == 2);
    assert(m1->elem[1] == 4);
    assert(m1->elem[2] == 6);
    assert(m1->elem[3] == 8);

    matrix_free(m);
    free(m);
    matrix_free(m1);
    free(m1);
}

void test_matrix_add_broadcasting_y_axis() {
    Matrix* m1 = matrix_init(NULL, 2, 2);
    m1->elem[0] = 1;
    m1->elem[1] = 3;
    m1->elem[2] = 5;
    m1->elem[3] = 7;

    Matrix* m2 = matrix_init(NULL, 2, 1);
    m2->elem[0] = 1;
    m2->elem[1] = 2;

    Matrix* m3 = matrix_add(m1, m2);
    assert(m3 != NULL);
    assert(m3->num_rows == 2);
    assert(m3->num_cols == 2);
    assert(m3->elem[0] == 2);
    assert(m3->elem[1] == 4);
    assert(m3->elem[2] == 7);
    assert(m3->elem[3] == 9);

    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    free(m1);
    free(m2);
    free(m3);
}

void test_matrix_add_broadcasting_x_axis() {
    Matrix* m1 = matrix_init(NULL, 2, 2);
    m1->elem[0] = 1;
    m1->elem[1] = 3;
    m1->elem[2] = 5;
    m1->elem[3] = 7;

    Matrix* m2 = matrix_init(NULL, 1, 2);
    m2->elem[0] = 1;
    m2->elem[1] = 2;

    Matrix* m3 = matrix_add(m1, m2);
    assert(m3 != NULL);
    assert(m3->num_rows == 2);
    assert(m3->num_cols == 2);
    printf("Hi there %f\n", m3->elem[0]);
    assert(m3->elem[0] == 2);
    assert(m3->elem[1] == 5);
    assert(m3->elem[2] == 6);
    assert(m3->elem[3] == 9);

    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    free(m1);
    free(m2);
    free(m3);
}

void test_matrix_add_broadcasting_both_axes() {
    Matrix* m1 = matrix_init(NULL, 2, 2);
    m1->elem[0] = 1;
    m1->elem[1] = 2;
    m1->elem[2] = 3;
    m1->elem[3] = 4;

    Matrix* m2 = matrix_init(NULL, 1, 1);
    m2->elem[0] = 1;

    Matrix* m3 = matrix_add(m1, m2);
    assert(m3 != NULL);
    assert(m3->num_rows == 2);
    assert(m3->num_cols == 2);
    assert(m3->elem[0] == 2);
    assert(m3->elem[1] == 3);
    assert(m3->elem[2] == 4);
    assert(m3->elem[3] == 5);

    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    free(m1);
    free(m2);
    free(m3);
}

void test_matrix_add_incompatible_dimensions() {
    // Doesn't bother intializing values, since our failure doesn't depend on them
    Matrix* m1 = matrix_init(NULL, 2, 2);

    Matrix* m2 = matrix_init(NULL, 3, 3);

    matrix_add(m1, m2);
}

void test_matrix_subtract_same_dimensions() {
    Matrix* m = matrix_init(NULL, 2, 2);
    m->elem[0] = 1;
    m->elem[1] = 2;
    m->elem[2] = 3;
    m->elem[3] = 4;

    // There shouldn't be problems with memory due to passing the same matrix
    Matrix* m1 = matrix_subtract(m, m);
    assert(m1 != NULL);
    assert(m1->num_rows == 2);
    assert(m1->num_cols == 2);

    DEBUG_PRINT(("elem 0 %f\n", m1->elem[0]));
    assert(m1->elem[0] == 0);
    assert(m1->elem[1] == 0);
    assert(m1->elem[2] == 0);
    assert(m1->elem[3] == 0);

    matrix_free(m);
    free(m);
    matrix_free(m1);
    free(m1);
}

void test_matrix_subtract_broadcasting_y_axis() {
    Matrix* m1 = matrix_init(NULL, 2, 2);
    m1->elem[0] = 1;
    m1->elem[1] = 3;
    m1->elem[2] = 5;
    m1->elem[3] = 7;

    Matrix* m2 = matrix_init(NULL, 2, 1);
    m2->elem[0] = 1;
    m2->elem[1] = 2;

    Matrix* m3 = matrix_subtract(m1, m2);
    assert(m3 != NULL);
    assert(m3->num_rows == 2);
    assert(m3->num_cols == 2);
    assert(m3->elem[0] == 0);
    assert(m3->elem[1] == 2);
    assert(m3->elem[2] == 3);
    assert(m3->elem[3] == 5);

    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    free(m1);
    free(m2);
    free(m3);
}

void test_matrix_subtract_broadcasting_x_axis() {
    Matrix* m1 = matrix_init(NULL, 2, 2);
    m1->elem[0] = 1;
    m1->elem[1] = 3;
    m1->elem[2] = 5;
    m1->elem[3] = 7;

    Matrix* m2 = matrix_init(NULL, 1, 2);
    m2->elem[0] = 1;
    m2->elem[1] = 2;

    Matrix* m3 = matrix_subtract(m1, m2);
    assert(m3 != NULL);
    assert(m3->num_rows == 2);
    assert(m3->num_cols == 2);
    assert(m3->elem[0] == 0);
    assert(m3->elem[1] == 1);
    assert(m3->elem[2] == 4);
    assert(m3->elem[3] == 5);

    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    free(m1);
    free(m2);
    free(m3);
}

void test_matrix_subtract_broadcasting_both_axes() {
    Matrix* m1 = matrix_init(NULL, 2, 2);
    m1->elem[0] = 1;
    m1->elem[1] = 2;
    m1->elem[2] = 3;
    m1->elem[3] = 4;

    Matrix* m2 = matrix_init(NULL, 1, 1);
    m2->elem[0] = 1;

    Matrix* m3 = matrix_subtract(m1, m2);
    assert(m3 != NULL);
    assert(m3->num_rows == 2);
    assert(m3->num_cols == 2);
    assert(m3->elem[0] == 0);
    assert(m3->elem[1] == 1);
    assert(m3->elem[2] == 2);
    assert(m3->elem[3] == 3);

    matrix_free(m1);
    matrix_free(m2);
    matrix_free(m3);
    free(m1);
    free(m2);
    free(m3);
}

void test_matrix_subtract_incompatible_dimensions() {
    // Doesn't bother intializing values, since our failure doesn't depend on them
    Matrix* m1 = matrix_init(NULL, 2, 2);
    Matrix* m2 = matrix_init(NULL, 3, 3);

    matrix_subtract(m1, m2);
}

void test_matrix_transpose() {
    Matrix* m = matrix_init(NULL, 3, 2);
    for (int i = 0; i < 6; i++) {
        m->elem[i] = (double)i + 1;
    }

    Matrix* trans = matrix_transpose(m);
    assert(trans->elem[0] == 1);
    assert(trans->elem[1] == 3);
    assert(trans->elem[2] == 5);
    assert(trans->elem[3] == 2);
    assert(trans->elem[4] == 4);
    assert(trans->elem[5] == 6);

    matrix_free(m);
    free(m);
    matrix_free(trans);
    free(trans);
}

void test_matrix_argmax() {
    Matrix* m = matrix_init(NULL, 2, 2);
    m->elem[0] = 0;
    m->elem[1] = 0;
    m->elem[2] = -1;
    m->elem[3] = 17;

    assert(matrix_argmax(m) == 3);
}

double minus_one(double a) {
    return a - 1;
}

void test_matrix_map() {
    Matrix* m = matrix_init(NULL, 2, 2);

    m->elem[0] = 0;
    m->elem[1] = 1;
    m->elem[2] = 2;
    m->elem[3] = 3;

    matrix_map_(m, &minus_one);

    assert(m->elem[0] == -1);
    assert(m->elem[1] == 0);
    assert(m->elem[2] == 1);
    assert(m->elem[3] == 2);

    matrix_sigmoid_(m);

    assert(m->elem[0] - 0.269 < 0.01); // sigmoid(-1)
    assert(m->elem[1] - 0.5 < 0.01);   // sigmoid(0)
    assert(m->elem[2] - 0.731 < 0.01); // sigmoid(1)
    assert(m->elem[3] - 0.881 < 0.01); // sigmoid(2)

    matrix_free(m);
    free(m);
}

void test_matrix_sigmoid() {
    Matrix* m1 = matrix_init(NULL, 2, 2);
    for (int i = 0; i < 4; i++) {
        m1->elem[i] = (double) i + 1;
    }

    Matrix* m2 = matrix_init_from(NULL, m1);

    matrix_sigmoid_(m1);

    assert(m1->elem[0] - 0.731 < 0.01);
    assert(m1->elem[1] - 0.881 < 0.01);
    assert(m1->elem[2] - 0.953 < 0.01);
    assert(m1->elem[3] - 0.982 < 0.01);

    matrix_sigmoid_prime_(m2);

    assert(m2->elem[0] - 0.197 < 0.01);
    assert(m2->elem[1] - 0.105 < 0.01);
    assert(m2->elem[2] - 0.045 < 0.01);
    assert(m2->elem[3] - 0.018 < 0.01);

    matrix_free(m1);
    matrix_free(m2);
    free(m1);
    free(m2);
}

void test_feed_forward_dimensions() {
    // This ends up being owned by the network
    int sizes[3];

    sizes[0] = 1;
    sizes[1] = 2;
    sizes[2] = 3;

    Network* net = create_network(3, sizes);

    Matrix* inp = matrix_init(NULL, 1, 1);

    inp->elem[0] = 1;

    Matrix* out = feed_forward(net, inp);

    // No assertions, just check if this runs

    free(out);
    free_network(net);
}

void test_feed_forward() {
    // This ends up being owned by the network
    int sizes[3];

    sizes[0] = 1;
    sizes[1] = 2;
    sizes[2] = 3;

    Network* net = create_network(3, sizes);

    // Make all weights and biases one
    for (int i = 0; i < net->num_layers; i++) {
        for (int j = 0; j < net->weights[i].num_rows * net->weights[i].num_cols; j++) {
            net->weights[i].elem[j] = 1;
        }

        for (int j = 0; j < net->biases[i].num_rows * net->biases[i].num_cols; j++) {
            net->biases[i].elem[j] = 1;
        }
    }

    Matrix* inp = matrix_init(NULL, 1, 1);

    inp->elem[0] = 1;

    Matrix* out = feed_forward(net, inp);

    assert(out != NULL);
    assert(out->num_rows == 3);
    assert(out->num_cols == 1);

    assert(out->elem[0] - 0.9506 < 0.01);
    assert(out->elem[1] - 0.9506 < 0.01);
    assert(out->elem[2] - 0.9506 < 0.01);

    free(out);
    free_network(net);
}

// Make sure nothing weird happens with the implicit casting
void test_image_to_matrix() {
    MnistImage img;
    for (int i = 0; i < 784; i++) {
        img.pixels[i] = (double)i;
    }

    Matrix* m = image_to_matrix(img);
    for (int i = 0; i < 784; i++) {
        assert(m->elem[i] - i < 0.01);
    }

    matrix_free(m);
    free(m);
}

// These actually start at 0, so there's no shifting
void test_label_to_matrix() {
    MnistLabel label = 9;
    Matrix* m = label_to_matrix(label);

    for (int i = 0; i < 9; i++) {
        assert(m->elem[i] == 0);
    }
    assert(m->elem[9] == 1);

    matrix_free(m);
    free(m);
}

// Run, expecting a exit failure
void run_return(void (*test_fn)(), int expected_return) {
    pid_t child_pid;
    Dl_info info;

    int filedes[2];
    if (pipe(filedes) == -1) {
        perror("Failed while setting up pipe");
        exit(1);
    }

    int result = dladdr((const void*)test_fn, &info);
    if (result == 0) {
        perror("Failed to find test function name, aborting!");
        exit(1);
    }

    if ((child_pid = fork()) == 0) {
        while ((dup2(filedes[1], STDOUT_FILENO) == -1) && (errno == EINTR));
        close(filedes[1]);
        close(filedes[0]);

        test_fn();
        exit(0);
    } else if (child_pid == -1) {
        perror("Failed while forking child to run test");
        exit(1);
    }
    close(filedes[1]);

    int status = 0;
    // Wait for the child process to terminate
    if (waitpid(child_pid, &status, 0) == -1) {
        perror("waitpid() failed");
        exit(EXIT_FAILURE);
    }
    int exit_status;
    if (WIFEXITED(status)) {
        exit_status = WEXITSTATUS(status);
    } else {
        exit_status = 1; // Declare failure if the process didn't exit normally
    }

    printf("Test %s %s!\n", info.dli_sname,
            exit_status == expected_return ? "succeded" : "failed");

    if (exit_status != expected_return || TESTVERBOSE) {
        printf("\tExpected return code %d, got %d\n", expected_return, exit_status);

        char output_buffer[4096];
        while (true) {
            ssize_t count = read(filedes[0], output_buffer, sizeof(output_buffer));
            if (count == -1) {
                if (errno == EINTR) {
                    continue;
                } else {
                    perror("An error occured while reading from the failed child's stdout");
                    exit(1);
                }
            } else if (count == 0) {
                break;
            } else {
                printf("%s\n", output_buffer);
            }
        }
    }

    close(filedes[0]);
}

void run(void (*test_fn)()) {
    run_return(test_fn, 0);
}

int main () {
    puts("Running tests");

    run(&test_mnist_loader);
    run(&test_network_init);
    run(&test_matrix_init_from);
    run(&test_matrix_init_zeros);

    run(&test_matrix_inner_product);
    run_return(&test_matrix_inner_product_fail, 1);
    run(&test_matrix_dot);
    run(&test_matrix_dot_2);
    run(&test_matrix_dot_under_aliasing);
    run(&test_matrix_times_scalar_right);
    run(&test_matrix_times_scalar_left);

    run(&test_matrix_hadamard_product);
    run(&test_matrix_hadamard_product_broadcasting);

    run(&test_matrix_add_same_dimensions);
    run(&test_matrix_add_broadcasting_x_axis);
    run(&test_matrix_add_broadcasting_y_axis);
    run(&test_matrix_add_broadcasting_both_axes);
    run_return(&test_matrix_add_incompatible_dimensions, 1);

    run(&test_matrix_subtract_same_dimensions);
    run(&test_matrix_subtract_broadcasting_x_axis);
    run(&test_matrix_subtract_broadcasting_y_axis);
    run(&test_matrix_subtract_broadcasting_both_axes);
    run_return(&test_matrix_subtract_incompatible_dimensions, 1);

    run(&test_matrix_transpose);
    run(&test_matrix_argmax);

    run(&test_matrix_map);
    run(&test_matrix_sigmoid);
    run(&test_feed_forward_dimensions);
    run(&test_feed_forward);
    run(&test_image_to_matrix);
    run(&test_label_to_matrix);
}
