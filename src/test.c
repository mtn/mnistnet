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

void test_matrix_inner_product() {
    Matrix* m1 = matrix_init(NULL, 1, 3);
    m1->elem[0] = 1;
    m1->elem[1] = 1;
    m1->elem[2] = 1;

    Matrix* m2 = matrix_init(NULL, 3, 1);
    m2->elem[0] = 1;
    m2->elem[1] = 1;
    m2->elem[2] = 1;

    Matrix* m3 = matrix_multiply(m1, m2);

    assert(m3->num_cols = 1);
    assert(m3->num_rows = 1);
    assert(m3->elem[0] == 3);
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
    matrix_multiply(m1, m2);
}

void test_matrix_multiply() {
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
    assert(m1->elem[0] = 2);
    assert(m1->elem[1] = 4);
    assert(m1->elem[2] = 6);
    assert(m1->elem[3] = 8);
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
    assert(m3->elem[0] = 2);
    assert(m3->elem[1] = 5);
    assert(m3->elem[2] = 6);
    assert(m3->elem[3] = 9);
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
    assert(m3->elem[0] = 1);
    assert(m3->elem[1] = 4);
    assert(m3->elem[2] = 7);
    assert(m3->elem[3] = 9);
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
    assert(m3->elem[0] = 2);
    assert(m3->elem[1] = 3);
    assert(m3->elem[2] = 4);
    assert(m3->elem[3] = 5);
}

void test_matrix_add_incompatible_dimensions() {
    // Doesn't bother intializing values, since our failure doesn't depend on them
    Matrix* m1 = matrix_init(NULL, 2, 2);

    Matrix* m2 = matrix_init(NULL, 3, 3);

    matrix_add(m1, m2);
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
}

void test_feed_forward() {
    int* sizes = malloc(sizeof(int) * 3);

    sizes[0] = 1;
    sizes[1] = 2;
    sizes[2] = 3;

    Network* net = create_network(3, sizes);

    Matrix* inp = matrix_init(NULL, 1, 1);

   inp->elem[0] = 1;
    /* inp->elem[2] = 1; */
    /* inp->elem[3] = 1; */

    Matrix* out = feed_forward(net, inp);

    // No assertions, just check if this runs

    free(out);
    free_network(net);
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
    waitpid(child_pid, &status, 0);
    int exit_status = WEXITSTATUS(status);

    printf("Test %s %s!\n", info.dli_sname,
            exit_status == expected_return ? "succeded" : "failed");

    if (exit_status != expected_return || TESTVERBOSE) {
        printf("Expected return code %d, got %d\n", expected_return, exit_status);

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
                printf("%s", output_buffer);
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
    run(&test_matrix_inner_product);
    run_return(&test_matrix_inner_product_fail, 1);
    run(&test_matrix_multiply);
    run(&test_matrix_times_scalar_right);
    run(&test_matrix_times_scalar_left);
    run(&test_matrix_add_same_dimensions);
    run(&test_matrix_add_broadcasting_x_axis);
    run(&test_matrix_add_broadcasting_y_axis);
    run(&test_matrix_add_broadcasting_both_axes);
    run_return(&test_matrix_add_incompatible_dimensions, 1);
    run(&test_matrix_map);
    run(&test_feed_forward);
}
