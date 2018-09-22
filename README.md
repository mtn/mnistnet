# mnistnet

mnistnet is a neural network from scratch in C that classifies handwritten digits from the [MNIST data set](http://yann.lecun.com/exdb/mnist/). I've worked on it while reading Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html). The goal was maximal learning!

Aside from standard C99 libraries, there are no external dependencies. Everything, from linear algebra to unit testing, is implemented from scratch. Of course, the design of these implementations draws heavy inspiration from established libraries like `numpy`. In particular, the linear algebra implementations are completely naive, and are much slower than their `numpy` counterparts.

## Usage

The `makefile` in the top-level directory includes most options. To build and train over 10 epochs, run `make run` or `make`. To build the binary, run `make mnistnet`. Because the random initialization is seeded, you'll see exactly this:

```
Epoch: 0
        Updating mini batches {1 - 1000} / 5000
        Updating mini batches {1001 - 2000} / 5000
        Updating mini batches {2001 - 3000} / 5000
        Updating mini batches {3001 - 4000} / 5000
        Updating mini batches {4001 - 5000} / 5000
Epoch 0: 9074 / 10000
Epoch: 1
        Updating mini batches {1 - 1000} / 5000
        Updating mini batches {1001 - 2000} / 5000
        Updating mini batches {2001 - 3000} / 5000
        Updating mini batches {3001 - 4000} / 5000
        Updating mini batches {4001 - 5000} / 5000
Epoch 1: 9277 / 10000
Epoch: 2
        Updating mini batches {1 - 1000} / 5000
        Updating mini batches {1001 - 2000} / 5000
        Updating mini batches {2001 - 3000} / 5000
        Updating mini batches {3001 - 4000} / 5000
        Updating mini batches {4001 - 5000} / 5000
Epoch 2: 9303 / 10000
Epoch: 3
        Updating mini batches {1 - 1000} / 5000
        Updating mini batches {1001 - 2000} / 5000
        Updating mini batches {2001 - 3000} / 5000
        Updating mini batches {3001 - 4000} / 5000
        Updating mini batches {4001 - 5000} / 5000
Epoch 3: 9405 / 10000
Epoch: 4
        Updating mini batches {1 - 1000} / 5000
        Updating mini batches {1001 - 2000} / 5000
        Updating mini batches {2001 - 3000} / 5000
        Updating mini batches {3001 - 4000} / 5000
        Updating mini batches {4001 - 5000} / 5000
Epoch 4: 9369 / 10000
Epoch: 5
        Updating mini batches {1 - 1000} / 5000
        Updating mini batches {1001 - 2000} / 5000
        Updating mini batches {2001 - 3000} / 5000
        Updating mini batches {3001 - 4000} / 5000
        Updating mini batches {4001 - 5000} / 5000
Epoch 5: 9407 / 10000
Epoch: 6
        Updating mini batches {1 - 1000} / 5000
        Updating mini batches {1001 - 2000} / 5000
        Updating mini batches {2001 - 3000} / 5000
        Updating mini batches {3001 - 4000} / 5000
        Updating mini batches {4001 - 5000} / 5000
Epoch 6: 9425 / 10000
Epoch: 7
        Updating mini batches {1 - 1000} / 5000
        Updating mini batches {1001 - 2000} / 5000
        Updating mini batches {2001 - 3000} / 5000
        Updating mini batches {3001 - 4000} / 5000
        Updating mini batches {4001 - 5000} / 5000
Epoch 7: 9451 / 10000
Epoch: 8
        Updating mini batches {1 - 1000} / 5000
        Updating mini batches {1001 - 2000} / 5000
        Updating mini batches {2001 - 3000} / 5000
        Updating mini batches {3001 - 4000} / 5000
        Updating mini batches {4001 - 5000} / 5000
Epoch 8: 9428 / 10000
Epoch: 9
        Updating mini batches {1 - 1000} / 5000
        Updating mini batches {1001 - 2000} / 5000
        Updating mini batches {2001 - 3000} / 5000
        Updating mini batches {3001 - 4000} / 5000
        Updating mini batches {4001 - 5000} / 5000
Epoch 9: 9470 / 10000
```
