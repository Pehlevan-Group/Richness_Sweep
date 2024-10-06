# Richness Sweep
Sweeping over the feature learning strength


## Datasets 

We create the MNIST-1M datasets using the script in the MNIST-1M folder.

Beyond this use CIFAR-10, [CIFAR-5M](https://github.com/preetum/cifar5m), and TinyImageNet datasets.

## Models 

The key MLP and CNN scripts are found in the vision_tasks directory. The vision_tasks.py script executes sweeps over various hyperparameters:

* N: width, ie number of neurons in the hidden layer
* L: depth, ie the number of 
* B: batch size for the optimizer
* s: noise level on the labels. We mostly don't sweep over this
* E: number of ensembles, ie copies of the network with different initialization seeds
* d: seed for the batch ordering. We mostly don't vary this
* task: MNIST-1M or CIFAR-5M for these tasks
* optimizer: Always SGD for now
* loss: Mean Squared Error (mse) or Cross-Entropy (xent)
* gamma0\_range, eta0\_range: the log10 range. E.g. 5 means the hyperparameter will vary from $10^{-5}$ to $10^5$
* gamma0\_res, eta0\_res: the number of trials per ten-folding. E.g. 2 means we run a sweep roughly every factor of 3
* range\_below\_max\_eta0: after the first convergent learning rate, how many factors of 10 down to sweep in eta
* save\_model: whether to save the weights of the model or not

