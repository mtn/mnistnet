#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include <errno.h>

#include "load_mnist.h"
#include "network.h"
#include "macros.h"
#include "nmath.h"
#include "util.h"


void test_mnist_loader() {
    FILE* fp = open_image_file("data/t10k-images-idx3-ubyte");

    read_image(fp);

    // No assertions, just check that this runs
    // Magic numbers are checked normally at runtime
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

void test_matrix_multiplication() {
    Matrix* m1 = malloc(sizeof(Matrix));
    matrix_init(m1, 2, 3);
    m1->elem[0] = 1;
    m1->elem[1] = 2;
    m1->elem[2] = 3;
    m1->elem[3] = 4;
    m1->elem[4] = 5;
    m1->elem[5] = 6;

    assert(m1 != NULL);
    assert(m1->num_rows == 2);
    assert(m1->num_cols == 3);

    Matrix* m2 = malloc(sizeof(Matrix));
    matrix_init(m2, 3, 2);
    m2->elem[0] = 7;
    m2->elem[1] = 8;
    m2->elem[2] = 9;
    m2->elem[3] = 10;
    m2->elem[4] = 11;
    m2->elem[5] = 12;

    assert(m2 != NULL);
    assert(m2->num_rows == 3);
    assert(m2->num_cols == 2);

    Matrix* m3 = matrix_multiply(m1, m2);

    assert(m3->num_rows == 2);
    assert(m3->num_cols == 2);
    assert(m3->elem[0] == 58);
    assert(m3->elem[1] == 64);
    assert(m3->elem[2] == 139);
    assert(m3->elem[3] == 154);
}

void test_feed_forward() {
    int* sizes = malloc(sizeof(int) * 3);

    sizes[0] = 3;
    sizes[1] = 2;
    sizes[2] = 3;

    Network* net = create_network(3, sizes);

    Matrix* inp = malloc(sizeof(Matrix));
    matrix_init(inp, 1, 3);

    inp->elem[0] = 1;
    inp->elem[2] = 1;
    inp->elem[3] = 1;

    Matrix* out = feed_forward(net, inp);

    // No assertions, just check if this runs

    free(out);
    free_network(net);
}

void run(void (*test_fn)()) {
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

    child_pid = fork();
    if (child_pid == 0) {
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

    printf("Test %s %s!\n", info.dli_sname,
            status == 0 ? "succeded" : "failed");

    // Dump output from stdout of failed processes
    if (status != 0) {
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

int main () {
    puts("Running tests");

    run(&test_mnist_loader);
    run(&test_network_init);
    run(&test_matrix_multiplication);
    run(&test_feed_forward);
}
