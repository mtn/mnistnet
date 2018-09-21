# mnistnet

mnistnet is a neural network from scratch in C that classifies handwritten digits from the [MNIST data set](http://yann.lecun.com/exdb/mnist/). I've worked on it while reading Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html). The goal was maximal learning!

Aside from standard C99 libraries, there are no external dependencies. Everything, from linear algebra to unit testing, is implemented from scratch. Of course, the design of these implementations draws heavy inspiration from established libraries like `numpy`. In particular, the linear algebra implementations are completely naive, and are much slower than their `numpy` counterparts.

## Usage

The `makefile` in the top-level directory includes most options. To build and train over 10 epochs, run `make run` or `make`. To build the binary, run `make mnistnet`.
