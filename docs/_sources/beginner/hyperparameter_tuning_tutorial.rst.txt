.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_beginner_hyperparameter_tuning_tutorial.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_beginner_hyperparameter_tuning_tutorial.py:


Hyperparameter tuning with Ray Tune
===================================

Hyperparameter tuning can make the difference between an average model and a highly
accurate one. Often simple things like choosing a different learning rate or changing
a network layer size can have a dramatic impact on your model performance.

Fortunately, there are tools that help with finding the best combination of parameters.
`Ray Tune <https://docs.ray.io/en/latest/tune.html>`_ is an industry standard tool for
distributed hyperparameter tuning. Ray Tune includes the latest hyperparameter search
algorithms, integrates with TensorBoard and other analysis libraries, and natively
supports distributed training through `Ray's distributed machine learning engine
<https://ray.io/>`_.

In this tutorial, we will show you how to integrate Ray Tune into your PyTorch
training workflow. We will extend `this tutorial from the PyTorch documentation
<https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_ for training
a CIFAR10 image classifier.

As you will see, we only need to add some slight modifications. In particular, we
need to

1. wrap data loading and training in functions,
2. make some network parameters configurable,
3. add checkpointing (optional),
4. and define the search space for the model tuning

|

To run this tutorial, please make sure the following packages are
installed:

-  ``ray[tune]``: Distributed hyperparameter tuning library
-  ``torchvision``: For the data transformers

Setup / Imports
---------------
Let's start with the imports:

.. code-block:: default

    from functools import partial
    import numpy as np
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import random_split
    import torchvision
    import torchvision.transforms as transforms
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler







Most of the imports are needed for building the PyTorch model. Only the last three
imports are for Ray Tune.

Data loaders
------------
We wrap the data loaders in their own function and pass a global data directory.
This way we can share a data directory between different trials.


.. code-block:: default



    def load_data(data_dir="./data"):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)

        return trainset, testset







Configurable neural network
---------------------------
We can only tune those parameters that are configurable. In this example, we can specify
the layer sizes of the fully connected layers:


.. code-block:: default



    class Net(nn.Module):
        def __init__(self, l1=120, l2=84):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, l1)
            self.fc2 = nn.Linear(l1, l2)
            self.fc3 = nn.Linear(l2, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x







The train function
------------------
Now it gets interesting, because we introduce some changes to the example `from the PyTorch
documentation <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_.

We wrap the training script in a function ``train_cifar(config, checkpoint_dir=None, data_dir=None)``.
As you can guess, the ``config`` parameter will receive the hyperparameters we would like to
train with. The ``checkpoint_dir`` parameter is used to restore checkpoints. The ``data_dir`` specifies
the directory where we load and store the data, so multiple runs can share the same data source.

.. code-block:: python

    net = Net(config["l1"], config["l2"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

The learning rate of the optimizer is made configurable, too:

.. code-block:: python

    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

We also split the training data into a training and validation subset. We thus train on
80% of the data and calculate the validation loss on the remaining 20%. The batch sizes
with which we iterate through the training and test sets are configurable as well.

Adding (multi) GPU support with DataParallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Image classification benefits largely from GPUs. Luckily, we can continue to use
PyTorch's abstractions in Ray Tune. Thus, we can wrap our model in ``nn.DataParallel``
to support data parallel training on multiple GPUs:

.. code-block:: python

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

By using a ``device`` variable we make sure that training also works when we have
no GPUs available. PyTorch requires us to send our data to the GPU memory explicitly,
like this:

.. code-block:: python

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

The code now supports training on CPUs, on a single GPU, and on multiple GPUs. Notably, Ray
also supports `fractional GPUs <https://docs.ray.io/en/master/using-ray-with-gpus.html#fractional-gpus>`_
so we can share GPUs among trials, as long as the model still fits on the GPU memory. We'll come back
to that later.

Communicating with Ray Tune
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most interesting part is the communication with Ray Tune:

.. code-block:: python

    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((net.state_dict(), optimizer.state_dict()), path)

    tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

Here we first save a checkpoint and then report some metrics back to Ray Tune. Specifically,
we send the validation loss and accuracy back to Ray Tune. Ray Tune can then use these metrics
to decide which hyperparameter configuration lead to the best results. These metrics
can also be used to stop bad performing trials early in order to avoid wasting
resources on those trials.

The checkpoint saving is optional, however, it is necessary if we wanted to use advanced
schedulers like
`Population Based Training <https://docs.ray.io/en/master/tune/tutorials/tune-advanced-tutorial.html>`_.
Also, by saving the checkpoint we can later load the trained models and validate them
on a test set.

Full training function
~~~~~~~~~~~~~~~~~~~~~~

The full code example looks like this:


.. code-block:: default



    def train_cifar(config, checkpoint_dir=None, data_dir=None):
        net = Net(config["l1"], config["l2"])

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        trainset, testset = load_data(data_dir)

        test_abs = int(len(trainset) * 0.8)
        train_subset, val_subset = random_split(
            trainset, [test_abs, len(trainset) - test_abs])

        trainloader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=8)
        valloader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=8)

        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    running_loss / epoch_steps))
                    running_loss = 0.0

            # Validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)

            tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
        print("Finished Training")







As you can see, most of the code is adapted directly from the original example.

Test set accuracy
-----------------
Commonly the performance of a machine learning model is tested on a hold-out test
set with data that has not been used for training the model. We also wrap this in a
function:


.. code-block:: default



    def test_accuracy(net, device="cpu"):
        trainset, testset = load_data()

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=2)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total







The function also expects a ``device`` parameter, so we can do the
test set validation on a GPU.

Configuring the search space
----------------------------
Lastly, we need to define Ray Tune's search space. Here is an example:

.. code-block:: python

    config = {
        "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }

The ``tune.sample_from()`` function makes it possible to define your own sample
methods to obtain hyperparameters. In this example, the ``l1`` and ``l2`` parameters
should be powers of 2 between 4 and 256, so either 4, 8, 16, 32, 64, 128, or 256.
The ``lr`` (learning rate) should be uniformly sampled between 0.0001 and 0.1. Lastly,
the batch size is a choice between 2, 4, 8, and 16.

At each trial, Ray Tune will now randomly sample a combination of parameters from these
search spaces. It will then train a number of models in parallel and find the best
performing one among these. We also use the ``ASHAScheduler`` which will terminate bad
performing trials early.

We wrap the ``train_cifar`` function with ``functools.partial`` to set the constant
``data_dir`` parameter. We can also tell Ray Tune what resources should be
available for each trial:

.. code-block:: python

    gpus_per_trial = 2
    # ...
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True)

You can specify the number of CPUs, which are then available e.g.
to increase the ``num_workers`` of the PyTorch ``DataLoader`` instances. The selected
number of GPUs are made visible to PyTorch in each trial. Trials do not have access to
GPUs that haven't been requested for them - so you don't have to care about two trials
using the same set of resources.

Here we can also specify fractional GPUs, so something like ``gpus_per_trial=0.5`` is
completely valid. The trials will then share GPUs among each other.
You just have to make sure that the models still fit in the GPU memory.

After training the models, we will find the best performing one and load the trained
network from the checkpoint file. We then obtain the test set accuracy and report
everything by printing.

The full main function looks like this:


.. code-block:: default



    def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
        data_dir = os.path.abspath("./data")
        load_data(data_dir)
        config = {
            "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([2, 4, 8, 16])
        }
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "accuracy", "training_iteration"])
        result = tune.run(
            partial(train_cifar, data_dir=data_dir),
            resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter)

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))

        best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(
            best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)

        test_acc = test_accuracy(best_trained_model, device)
        print("Best trial test set accuracy: {}".format(test_acc))


    if __name__ == "__main__":
        # You can change the number of GPUs per trial here:
        main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Files already downloaded and verified
    Files already downloaded and verified
    == Status ==
    Memory usage on this node: 32.0/251.8 GiB
    Using AsyncHyperBand: num_stopped=0
    Bracket: Iter 8.000: None | Iter 4.000: None | Iter 2.000: None | Iter 1.000: None
    Resources requested: 2/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 1/10 (1 RUNNING)
    +---------------------+----------+-------+--------------+------+------+-----------+
    | Trial name          | status   | loc   |   batch_size |   l1 |   l2 |        lr |
    |---------------------+----------+-------+--------------+------+------+-----------|
    | DEFAULT_504de_00000 | RUNNING  |       |            4 |  128 |    8 | 0.0233167 |
    +---------------------+----------+-------+--------------+------+------+-----------+


    [2m[36m(pid=2642083)[0m Files already downloaded and verified
    [2m[36m(pid=2642086)[0m Files already downloaded and verified
    [2m[36m(pid=2642085)[0m Files already downloaded and verified
    [2m[36m(pid=2642082)[0m Files already downloaded and verified
    [2m[36m(pid=2642092)[0m Files already downloaded and verified
    [2m[36m(pid=2642084)[0m Files already downloaded and verified
    [2m[36m(pid=2642090)[0m Files already downloaded and verified
    [2m[36m(pid=2642087)[0m Files already downloaded and verified
    [2m[36m(pid=2642089)[0m Files already downloaded and verified
    [2m[36m(pid=2642088)[0m Files already downloaded and verified
    [2m[36m(pid=2642083)[0m Files already downloaded and verified
    [2m[36m(pid=2642092)[0m Files already downloaded and verified
    [2m[36m(pid=2642084)[0m Files already downloaded and verified
    [2m[36m(pid=2642090)[0m Files already downloaded and verified
    [2m[36m(pid=2642088)[0m Files already downloaded and verified
    [2m[36m(pid=2642085)[0m Files already downloaded and verified
    [2m[36m(pid=2642082)[0m Files already downloaded and verified
    [2m[36m(pid=2642086)[0m Files already downloaded and verified
    [2m[36m(pid=2642087)[0m Files already downloaded and verified
    [2m[36m(pid=2642089)[0m Files already downloaded and verified
    [2m[36m(pid=2642083)[0m [1,  2000] loss: 2.147
    [2m[36m(pid=2642086)[0m [1,  2000] loss: 2.212
    [2m[36m(pid=2642092)[0m [1,  2000] loss: 2.160
    [2m[36m(pid=2642087)[0m [1,  2000] loss: 2.317
    [2m[36m(pid=2642090)[0m [1,  2000] loss: 2.298
    [2m[36m(pid=2642084)[0m [1,  2000] loss: 2.293
    [2m[36m(pid=2642089)[0m [1,  2000] loss: 2.148
    [2m[36m(pid=2642088)[0m [1,  2000] loss: 2.344
    [2m[36m(pid=2642082)[0m [1,  2000] loss: 1.951
    [2m[36m(pid=2642085)[0m [1,  2000] loss: 2.033
    [2m[36m(pid=2642086)[0m [1,  4000] loss: 0.998
    [2m[36m(pid=2642083)[0m [1,  4000] loss: 0.919
    [2m[36m(pid=2642092)[0m [1,  4000] loss: 0.900
    [2m[36m(pid=2642090)[0m [1,  4000] loss: 1.156
    [2m[36m(pid=2642084)[0m [1,  4000] loss: 1.157
    [2m[36m(pid=2642087)[0m [1,  4000] loss: 1.157
    [2m[36m(pid=2642088)[0m [1,  4000] loss: 1.160
    [2m[36m(pid=2642089)[0m [1,  4000] loss: 1.124
    [2m[36m(pid=2642082)[0m [1,  4000] loss: 0.804
    [2m[36m(pid=2642085)[0m [1,  4000] loss: 0.811
    [2m[36m(pid=2642086)[0m [1,  6000] loss: 0.633
    [2m[36m(pid=2642083)[0m [1,  6000] loss: 0.572
    [2m[36m(pid=2642092)[0m [1,  6000] loss: 0.557
    Result for DEFAULT_504de_00007:
      accuracy: 0.0984
      date: 2021-04-17_09-31-39
      done: false
      experiment_id: 7adc784bd3e140a9a1bd3db81ac1e8f2
      hostname: devfair017
      iterations_since_restore: 1
      loss: 2.311030323410034
      node_ip: 100.97.17.135
      pid: 2642090
      should_checkpoint: true
      time_since_restore: 29.17088484764099
      time_this_iter_s: 29.17088484764099
      time_total_s: 29.17088484764099
      timestamp: 1618677099
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: 504de_00007
  
    == Status ==
    Memory usage on this node: 36.4/251.8 GiB
    Using AsyncHyperBand: num_stopped=0
    Bracket: Iter 8.000: None | Iter 4.000: None | Iter 2.000: None | Iter 1.000: -2.311030323410034
    Resources requested: 20/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (10 RUNNING)
    +---------------------+----------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status   | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+----------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00000 | RUNNING  |                       |            4 |  128 |    8 | 0.0233167   |         |            |                      |
    | DEFAULT_504de_00001 | RUNNING  |                       |            8 |  128 |   32 | 0.00531987  |         |            |                      |
    | DEFAULT_504de_00002 | RUNNING  |                       |            8 |   32 |    4 | 0.000197384 |         |            |                      |
    | DEFAULT_504de_00003 | RUNNING  |                       |            8 |    8 |  256 | 0.0184795   |         |            |                      |
    | DEFAULT_504de_00004 | RUNNING  |                       |            8 |  128 |  128 | 0.00233347  |         |            |                      |
    | DEFAULT_504de_00005 | RUNNING  |                       |            2 |   32 |   16 | 0.00280664  |         |            |                      |
    | DEFAULT_504de_00006 | RUNNING  |                       |            2 |   16 |  256 | 0.00111242  |         |            |                      |
    | DEFAULT_504de_00007 | RUNNING  | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31103 |     0.0984 |                    1 |
    | DEFAULT_504de_00008 | RUNNING  |                       |            8 |    8 |  128 | 0.0388454   |         |            |                      |
    | DEFAULT_504de_00009 | RUNNING  |                       |            4 |   16 |   16 | 0.00211468  |         |            |                      |
    +---------------------+----------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00008:
      accuracy: 0.1002
      date: 2021-04-17_09-31-39
      done: true
      experiment_id: d4a954009b89443184f90650a1058c92
      hostname: devfair017
      iterations_since_restore: 1
      loss: 2.3274890979766845
      node_ip: 100.97.17.135
      pid: 2642084
      should_checkpoint: true
      time_since_restore: 29.24080753326416
      time_this_iter_s: 29.24080753326416
      time_total_s: 29.24080753326416
      timestamp: 1618677099
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: 504de_00008
  
    Result for DEFAULT_504de_00002:
      accuracy: 0.1263
      date: 2021-04-17_09-31-39
      done: false
      experiment_id: da3434fb89c5469e87ceb3aad2c37783
      hostname: devfair017
      iterations_since_restore: 1
      loss: 2.306386479759216
      node_ip: 100.97.17.135
      pid: 2642088
      should_checkpoint: true
      time_since_restore: 29.321458101272583
      time_this_iter_s: 29.321458101272583
      time_total_s: 29.321458101272583
      timestamp: 1618677099
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: 504de_00002
  
    Result for DEFAULT_504de_00003:
      accuracy: 0.1134
      date: 2021-04-17_09-31-40
      done: false
      experiment_id: b745df8b21204cefb0a4ba1b24438d0a
      hostname: devfair017
      iterations_since_restore: 1
      loss: 2.2964809747695925
      node_ip: 100.97.17.135
      pid: 2642089
      should_checkpoint: true
      time_since_restore: 29.471698760986328
      time_this_iter_s: 29.471698760986328
      time_total_s: 29.471698760986328
      timestamp: 1618677100
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: 504de_00003
  
    [2m[36m(pid=2642086)[0m [1,  8000] loss: 0.456
    Result for DEFAULT_504de_00001:
      accuracy: 0.4511
      date: 2021-04-17_09-31-41
      done: false
      experiment_id: c44a3113e5784b0f83f85e6a34ce7cc5
      hostname: devfair017
      iterations_since_restore: 1
      loss: 1.5044937871932984
      node_ip: 100.97.17.135
      pid: 2642082
      should_checkpoint: true
      time_since_restore: 30.514458179473877
      time_this_iter_s: 30.514458179473877
      time_total_s: 30.514458179473877
      timestamp: 1618677101
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: 504de_00001
  
    [2m[36m(pid=2642083)[0m [1,  8000] loss: 0.420
    [2m[36m(pid=2642087)[0m [1,  6000] loss: 0.772
    Result for DEFAULT_504de_00004:
      accuracy: 0.472
      date: 2021-04-17_09-31-41
      done: false
      experiment_id: 5212d8943b3340e09d2704ffe1649c65
      hostname: devfair017
      iterations_since_restore: 1
      loss: 1.4559912428855897
      node_ip: 100.97.17.135
      pid: 2642085
      should_checkpoint: true
      time_since_restore: 31.26694631576538
      time_this_iter_s: 31.26694631576538
      time_total_s: 31.26694631576538
      timestamp: 1618677101
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: 504de_00004
  
    [2m[36m(pid=2642092)[0m [1,  8000] loss: 0.402
    [2m[36m(pid=2642086)[0m [1, 10000] loss: 0.362
    [2m[36m(pid=2642083)[0m [1, 10000] loss: 0.324
    [2m[36m(pid=2642090)[0m [2,  2000] loss: 2.310
    [2m[36m(pid=2642088)[0m [2,  2000] loss: 2.300
    [2m[36m(pid=2642089)[0m [2,  2000] loss: 2.336
    [2m[36m(pid=2642082)[0m [2,  2000] loss: 1.453
    [2m[36m(pid=2642087)[0m [1,  8000] loss: 0.579
    [2m[36m(pid=2642092)[0m [1, 10000] loss: 0.315
    [2m[36m(pid=2642085)[0m [2,  2000] loss: 1.437
    [2m[36m(pid=2642086)[0m [1, 12000] loss: 0.295
    [2m[36m(pid=2642083)[0m [1, 12000] loss: 0.268
    Result for DEFAULT_504de_00009:
      accuracy: 0.4495
      date: 2021-04-17_09-31-56
      done: false
      experiment_id: ff92523354ac40d1ba16343aa9c712d3
      hostname: devfair017
      iterations_since_restore: 1
      loss: 1.5184569564640522
      node_ip: 100.97.17.135
      pid: 2642092
      should_checkpoint: true
      time_since_restore: 45.6649534702301
      time_this_iter_s: 45.6649534702301
      time_total_s: 45.6649534702301
      timestamp: 1618677116
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: 504de_00009
  
    == Status ==
    Memory usage on this node: 36.0/251.8 GiB
    Using AsyncHyperBand: num_stopped=1
    Bracket: Iter 8.000: None | Iter 4.000: None | Iter 2.000: None | Iter 1.000: -2.2964809747695925
    Resources requested: 18/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (9 RUNNING, 1 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00000 | RUNNING    |                       |            4 |  128 |    8 | 0.0233167   |         |            |                      |
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.50449 |     0.4511 |                    1 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 2.30639 |     0.1263 |                    1 |
    | DEFAULT_504de_00003 | RUNNING    | 100.97.17.135:2642089 |            8 |    8 |  256 | 0.0184795   | 2.29648 |     0.1134 |                    1 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.45599 |     0.472  |                    1 |
    | DEFAULT_504de_00005 | RUNNING    |                       |            2 |   32 |   16 | 0.00280664  |         |            |                      |
    | DEFAULT_504de_00006 | RUNNING    |                       |            2 |   16 |  256 | 0.00111242  |         |            |                      |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31103 |     0.0984 |                    1 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.51846 |     0.4495 |                    1 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642090)[0m [2,  4000] loss: 1.155
    [2m[36m(pid=2642088)[0m [2,  4000] loss: 1.140
    [2m[36m(pid=2642089)[0m [2,  4000] loss: 1.155
    [2m[36m(pid=2642082)[0m [2,  4000] loss: 0.715
    [2m[36m(pid=2642087)[0m [1, 10000] loss: 0.463
    [2m[36m(pid=2642086)[0m [1, 14000] loss: 0.253
    [2m[36m(pid=2642085)[0m [2,  4000] loss: 0.688
    [2m[36m(pid=2642083)[0m [1, 14000] loss: 0.226
    [2m[36m(pid=2642092)[0m [2,  2000] loss: 1.510
    Result for DEFAULT_504de_00000:
      accuracy: 0.1002
      date: 2021-04-17_09-32-05
      done: true
      experiment_id: 2a3c22c8ade04791ad0464a569f66da9
      hostname: devfair017
      iterations_since_restore: 1
      loss: 2.3131334617614745
      node_ip: 100.97.17.135
      pid: 2642087
      should_checkpoint: true
      time_since_restore: 54.63522911071777
      time_this_iter_s: 54.63522911071777
      time_total_s: 54.63522911071777
      timestamp: 1618677125
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: 504de_00000
  
    == Status ==
    Memory usage on this node: 36.0/251.8 GiB
    Using AsyncHyperBand: num_stopped=2
    Bracket: Iter 8.000: None | Iter 4.000: None | Iter 2.000: None | Iter 1.000: -2.3014337272644045
    Resources requested: 18/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (9 RUNNING, 1 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00000 | RUNNING    | 100.97.17.135:2642087 |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.50449 |     0.4511 |                    1 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 2.30639 |     0.1263 |                    1 |
    | DEFAULT_504de_00003 | RUNNING    | 100.97.17.135:2642089 |            8 |    8 |  256 | 0.0184795   | 2.29648 |     0.1134 |                    1 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.45599 |     0.472  |                    1 |
    | DEFAULT_504de_00005 | RUNNING    |                       |            2 |   32 |   16 | 0.00280664  |         |            |                      |
    | DEFAULT_504de_00006 | RUNNING    |                       |            2 |   16 |  256 | 0.00111242  |         |            |                      |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31103 |     0.0984 |                    1 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.51846 |     0.4495 |                    1 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00007:
      accuracy: 0.1004
      date: 2021-04-17_09-32-05
      done: false
      experiment_id: 7adc784bd3e140a9a1bd3db81ac1e8f2
      hostname: devfair017
      iterations_since_restore: 2
      loss: 2.3083166694641113
      node_ip: 100.97.17.135
      pid: 2642090
      should_checkpoint: true
      time_since_restore: 55.07859539985657
      time_this_iter_s: 25.907710552215576
      time_total_s: 55.07859539985657
      timestamp: 1618677125
      timesteps_since_restore: 0
      training_iteration: 2
      trial_id: 504de_00007
  
    Result for DEFAULT_504de_00002:
      accuracy: 0.1498
      date: 2021-04-17_09-32-05
      done: false
      experiment_id: da3434fb89c5469e87ceb3aad2c37783
      hostname: devfair017
      iterations_since_restore: 2
      loss: 2.241386289691925
      node_ip: 100.97.17.135
      pid: 2642088
      should_checkpoint: true
      time_since_restore: 55.154112815856934
      time_this_iter_s: 25.83265471458435
      time_total_s: 55.154112815856934
      timestamp: 1618677125
      timesteps_since_restore: 0
      training_iteration: 2
      trial_id: 504de_00002
  
    Result for DEFAULT_504de_00003:
      accuracy: 0.1006
      date: 2021-04-17_09-32-06
      done: true
      experiment_id: b745df8b21204cefb0a4ba1b24438d0a
      hostname: devfair017
      iterations_since_restore: 2
      loss: 2.3044813566207885
      node_ip: 100.97.17.135
      pid: 2642089
      should_checkpoint: true
      time_since_restore: 55.791253328323364
      time_this_iter_s: 26.319554567337036
      time_total_s: 55.791253328323364
      timestamp: 1618677126
      timesteps_since_restore: 0
      training_iteration: 2
      trial_id: 504de_00003
  
    Result for DEFAULT_504de_00001:
      accuracy: 0.5155
      date: 2021-04-17_09-32-07
      done: false
      experiment_id: c44a3113e5784b0f83f85e6a34ce7cc5
      hostname: devfair017
      iterations_since_restore: 2
      loss: 1.3715872536182403
      node_ip: 100.97.17.135
      pid: 2642082
      should_checkpoint: true
      time_since_restore: 56.77698993682861
      time_this_iter_s: 26.262531757354736
      time_total_s: 56.77698993682861
      timestamp: 1618677127
      timesteps_since_restore: 0
      training_iteration: 2
      trial_id: 504de_00001
  
    [2m[36m(pid=2642086)[0m [1, 16000] loss: 0.222
    [2m[36m(pid=2642083)[0m [1, 16000] loss: 0.196
    Result for DEFAULT_504de_00004:
      accuracy: 0.4913
      date: 2021-04-17_09-32-08
      done: false
      experiment_id: 5212d8943b3340e09d2704ffe1649c65
      hostname: devfair017
      iterations_since_restore: 2
      loss: 1.443132883167267
      node_ip: 100.97.17.135
      pid: 2642085
      should_checkpoint: true
      time_since_restore: 58.14488410949707
      time_this_iter_s: 26.87793779373169
      time_total_s: 58.14488410949707
      timestamp: 1618677128
      timesteps_since_restore: 0
      training_iteration: 2
      trial_id: 504de_00004
  
    [2m[36m(pid=2642092)[0m [2,  4000] loss: 0.740
    [2m[36m(pid=2642086)[0m [1, 18000] loss: 0.195
    [2m[36m(pid=2642083)[0m [1, 18000] loss: 0.171
    [2m[36m(pid=2642090)[0m [3,  2000] loss: 2.311
    [2m[36m(pid=2642088)[0m [3,  2000] loss: 2.219
    [2m[36m(pid=2642082)[0m [3,  2000] loss: 1.332
    [2m[36m(pid=2642085)[0m [3,  2000] loss: 1.267
    [2m[36m(pid=2642092)[0m [2,  6000] loss: 0.494
    [2m[36m(pid=2642086)[0m [1, 20000] loss: 0.174
    [2m[36m(pid=2642083)[0m [1, 20000] loss: 0.150
    [2m[36m(pid=2642090)[0m [3,  4000] loss: 1.156
    [2m[36m(pid=2642088)[0m [3,  4000] loss: 1.071
    [2m[36m(pid=2642082)[0m [3,  4000] loss: 0.672
    [2m[36m(pid=2642092)[0m [2,  8000] loss: 0.363
    [2m[36m(pid=2642085)[0m [3,  4000] loss: 0.623
    Result for DEFAULT_504de_00005:
      accuracy: 0.3313
      date: 2021-04-17_09-32-29
      done: false
      experiment_id: adee13b093d94872921b71add63a3059
      hostname: devfair017
      iterations_since_restore: 1
      loss: 1.8246950269937516
      node_ip: 100.97.17.135
      pid: 2642086
      should_checkpoint: true
      time_since_restore: 78.69783592224121
      time_this_iter_s: 78.69783592224121
      time_total_s: 78.69783592224121
      timestamp: 1618677149
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: 504de_00005
  
    == Status ==
    Memory usage on this node: 35.2/251.8 GiB
    Using AsyncHyperBand: num_stopped=3
    Bracket: Iter 8.000: None | Iter 4.000: None | Iter 2.000: -2.241386289691925 | Iter 1.000: -2.2964809747695925
    Resources requested: 14/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (7 RUNNING, 3 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.37159 |     0.5155 |                    2 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 2.24139 |     0.1498 |                    2 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.44313 |     0.4913 |                    2 |
    | DEFAULT_504de_00005 | RUNNING    | 100.97.17.135:2642086 |            2 |   32 |   16 | 0.00280664  | 1.8247  |     0.3313 |                    1 |
    | DEFAULT_504de_00006 | RUNNING    |                       |            2 |   16 |  256 | 0.00111242  |         |            |                      |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.30832 |     0.1004 |                    2 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.51846 |     0.4495 |                    1 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00006:
      accuracy: 0.4203
      date: 2021-04-17_09-32-29
      done: false
      experiment_id: 0ca109a83c054e1a9950179524f91481
      hostname: devfair017
      iterations_since_restore: 1
      loss: 1.5391802021205425
      node_ip: 100.97.17.135
      pid: 2642083
      should_checkpoint: true
      time_since_restore: 79.06968927383423
      time_this_iter_s: 79.06968927383423
      time_total_s: 79.06968927383423
      timestamp: 1618677149
      timesteps_since_restore: 0
      training_iteration: 1
      trial_id: 504de_00006
  
    Result for DEFAULT_504de_00007:
      accuracy: 0.0984
      date: 2021-04-17_09-32-30
      done: false
      experiment_id: 7adc784bd3e140a9a1bd3db81ac1e8f2
      hostname: devfair017
      iterations_since_restore: 3
      loss: 2.3078102285385134
      node_ip: 100.97.17.135
      pid: 2642090
      should_checkpoint: true
      time_since_restore: 79.78852415084839
      time_this_iter_s: 24.70992875099182
      time_total_s: 79.78852415084839
      timestamp: 1618677150
      timesteps_since_restore: 0
      training_iteration: 3
      trial_id: 504de_00007
  
    Result for DEFAULT_504de_00002:
      accuracy: 0.2128
      date: 2021-04-17_09-32-30
      done: false
      experiment_id: da3434fb89c5469e87ceb3aad2c37783
      hostname: devfair017
      iterations_since_restore: 3
      loss: 2.02995011138916
      node_ip: 100.97.17.135
      pid: 2642088
      should_checkpoint: true
      time_since_restore: 80.30428957939148
      time_this_iter_s: 25.150176763534546
      time_total_s: 80.30428957939148
      timestamp: 1618677150
      timesteps_since_restore: 0
      training_iteration: 3
      trial_id: 504de_00002
  
    Result for DEFAULT_504de_00001:
      accuracy: 0.5064
      date: 2021-04-17_09-32-32
      done: false
      experiment_id: c44a3113e5784b0f83f85e6a34ce7cc5
      hostname: devfair017
      iterations_since_restore: 3
      loss: 1.4183358983755112
      node_ip: 100.97.17.135
      pid: 2642082
      should_checkpoint: true
      time_since_restore: 81.92934083938599
      time_this_iter_s: 25.152350902557373
      time_total_s: 81.92934083938599
      timestamp: 1618677152
      timesteps_since_restore: 0
      training_iteration: 3
      trial_id: 504de_00001
  
    [2m[36m(pid=2642092)[0m [2, 10000] loss: 0.288
    Result for DEFAULT_504de_00004:
      accuracy: 0.5625
      date: 2021-04-17_09-32-34
      done: false
      experiment_id: 5212d8943b3340e09d2704ffe1649c65
      hostname: devfair017
      iterations_since_restore: 3
      loss: 1.2542812789440154
      node_ip: 100.97.17.135
      pid: 2642085
      should_checkpoint: true
      time_since_restore: 84.2343397140503
      time_this_iter_s: 26.089455604553223
      time_total_s: 84.2343397140503
      timestamp: 1618677154
      timesteps_since_restore: 0
      training_iteration: 3
      trial_id: 504de_00004
  
    == Status ==
    Memory usage on this node: 35.3/251.8 GiB
    Using AsyncHyperBand: num_stopped=3
    Bracket: Iter 8.000: None | Iter 4.000: None | Iter 2.000: -2.241386289691925 | Iter 1.000: -2.060588000881672
    Resources requested: 14/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (7 RUNNING, 3 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.41834 |     0.5064 |                    3 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 2.02995 |     0.2128 |                    3 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.25428 |     0.5625 |                    3 |
    | DEFAULT_504de_00005 | RUNNING    | 100.97.17.135:2642086 |            2 |   32 |   16 | 0.00280664  | 1.8247  |     0.3313 |                    1 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.53918 |     0.4203 |                    1 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.30781 |     0.0984 |                    3 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.51846 |     0.4495 |                    1 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642086)[0m [2,  2000] loss: 1.719
    [2m[36m(pid=2642083)[0m [2,  2000] loss: 1.468
    Result for DEFAULT_504de_00009:
      accuracy: 0.5003
      date: 2021-04-17_09-32-37
      done: false
      experiment_id: ff92523354ac40d1ba16343aa9c712d3
      hostname: devfair017
      iterations_since_restore: 2
      loss: 1.3994132188498973
      node_ip: 100.97.17.135
      pid: 2642092
      should_checkpoint: true
      time_since_restore: 87.18250846862793
      time_this_iter_s: 41.51755499839783
      time_total_s: 87.18250846862793
      timestamp: 1618677157
      timesteps_since_restore: 0
      training_iteration: 2
      trial_id: 504de_00009
  
    [2m[36m(pid=2642090)[0m [4,  2000] loss: 2.311
    [2m[36m(pid=2642088)[0m [4,  2000] loss: 1.978
    [2m[36m(pid=2642082)[0m [4,  2000] loss: 1.261
    [2m[36m(pid=2642086)[0m [2,  4000] loss: 0.866
    [2m[36m(pid=2642083)[0m [2,  4000] loss: 0.740
    [2m[36m(pid=2642085)[0m [4,  2000] loss: 1.148
    [2m[36m(pid=2642092)[0m [3,  2000] loss: 1.398
    [2m[36m(pid=2642090)[0m [4,  4000] loss: 1.156
    [2m[36m(pid=2642086)[0m [2,  6000] loss: 0.571
    [2m[36m(pid=2642088)[0m [4,  4000] loss: 0.962
    [2m[36m(pid=2642083)[0m [2,  6000] loss: 0.488
    [2m[36m(pid=2642082)[0m [4,  4000] loss: 0.653
    [2m[36m(pid=2642092)[0m [3,  4000] loss: 0.704
    [2m[36m(pid=2642085)[0m [4,  4000] loss: 0.582
    [2m[36m(pid=2642086)[0m [2,  8000] loss: 0.430
    [2m[36m(pid=2642083)[0m [2,  8000] loss: 0.368
    Result for DEFAULT_504de_00007:
      accuracy: 0.1013
      date: 2021-04-17_09-32-55
      done: false
      experiment_id: 7adc784bd3e140a9a1bd3db81ac1e8f2
      hostname: devfair017
      iterations_since_restore: 4
      loss: 2.3177268993377687
      node_ip: 100.97.17.135
      pid: 2642090
      should_checkpoint: true
      time_since_restore: 104.72373867034912
      time_this_iter_s: 24.935214519500732
      time_total_s: 104.72373867034912
      timestamp: 1618677175
      timesteps_since_restore: 0
      training_iteration: 4
      trial_id: 504de_00007
  
    == Status ==
    Memory usage on this node: 35.4/251.8 GiB
    Using AsyncHyperBand: num_stopped=3
    Bracket: Iter 8.000: None | Iter 4.000: -2.3177268993377687 | Iter 2.000: -1.8422595864295959 | Iter 1.000: -2.060588000881672
    Resources requested: 14/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (7 RUNNING, 3 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.41834 |     0.5064 |                    3 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 2.02995 |     0.2128 |                    3 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.25428 |     0.5625 |                    3 |
    | DEFAULT_504de_00005 | RUNNING    | 100.97.17.135:2642086 |            2 |   32 |   16 | 0.00280664  | 1.8247  |     0.3313 |                    1 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.53918 |     0.4203 |                    1 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31773 |     0.1013 |                    4 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.39941 |     0.5003 |                    2 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00002:
      accuracy: 0.2646
      date: 2021-04-17_09-32-55
      done: false
      experiment_id: da3434fb89c5469e87ceb3aad2c37783
      hostname: devfair017
      iterations_since_restore: 4
      loss: 1.86079810590744
      node_ip: 100.97.17.135
      pid: 2642088
      should_checkpoint: true
      time_since_restore: 105.2481541633606
      time_this_iter_s: 24.943864583969116
      time_total_s: 105.2481541633606
      timestamp: 1618677175
      timesteps_since_restore: 0
      training_iteration: 4
      trial_id: 504de_00002
  
    Result for DEFAULT_504de_00001:
      accuracy: 0.5122
      date: 2021-04-17_09-32-58
      done: false
      experiment_id: c44a3113e5784b0f83f85e6a34ce7cc5
      hostname: devfair017
      iterations_since_restore: 4
      loss: 1.3869041404962539
      node_ip: 100.97.17.135
      pid: 2642082
      should_checkpoint: true
      time_since_restore: 107.4989402294159
      time_this_iter_s: 25.569599390029907
      time_total_s: 107.4989402294159
      timestamp: 1618677178
      timesteps_since_restore: 0
      training_iteration: 4
      trial_id: 504de_00001
  
    [2m[36m(pid=2642092)[0m [3,  6000] loss: 0.459
    Result for DEFAULT_504de_00004:
      accuracy: 0.5767
      date: 2021-04-17_09-33-00
      done: false
      experiment_id: 5212d8943b3340e09d2704ffe1649c65
      hostname: devfair017
      iterations_since_restore: 4
      loss: 1.1873548250555992
      node_ip: 100.97.17.135
      pid: 2642085
      should_checkpoint: true
      time_since_restore: 110.29630422592163
      time_this_iter_s: 26.061964511871338
      time_total_s: 110.29630422592163
      timestamp: 1618677180
      timesteps_since_restore: 0
      training_iteration: 4
      trial_id: 504de_00004
  
    == Status ==
    Memory usage on this node: 35.4/251.8 GiB
    Using AsyncHyperBand: num_stopped=3
    Bracket: Iter 8.000: None | Iter 4.000: -1.623851123201847 | Iter 2.000: -1.8422595864295959 | Iter 1.000: -2.060588000881672
    Resources requested: 14/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (7 RUNNING, 3 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.3869  |     0.5122 |                    4 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.8608  |     0.2646 |                    4 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.18735 |     0.5767 |                    4 |
    | DEFAULT_504de_00005 | RUNNING    | 100.97.17.135:2642086 |            2 |   32 |   16 | 0.00280664  | 1.8247  |     0.3313 |                    1 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.53918 |     0.4203 |                    1 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31773 |     0.1013 |                    4 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.39941 |     0.5003 |                    2 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642086)[0m [2, 10000] loss: 0.342
    [2m[36m(pid=2642083)[0m [2, 10000] loss: 0.288
    [2m[36m(pid=2642090)[0m [5,  2000] loss: 2.312
    [2m[36m(pid=2642088)[0m [5,  2000] loss: 1.837
    [2m[36m(pid=2642092)[0m [3,  8000] loss: 0.346
    [2m[36m(pid=2642082)[0m [5,  2000] loss: 1.236
    [2m[36m(pid=2642086)[0m [2, 12000] loss: 0.289
    [2m[36m(pid=2642083)[0m [2, 12000] loss: 0.241
    [2m[36m(pid=2642085)[0m [5,  2000] loss: 1.069
    [2m[36m(pid=2642092)[0m [3, 10000] loss: 0.277
    [2m[36m(pid=2642090)[0m [5,  4000] loss: 1.156
    [2m[36m(pid=2642088)[0m [5,  4000] loss: 0.897
    [2m[36m(pid=2642086)[0m [2, 14000] loss: 0.246
    [2m[36m(pid=2642083)[0m [2, 14000] loss: 0.207
    [2m[36m(pid=2642082)[0m [5,  4000] loss: 0.621
    Result for DEFAULT_504de_00009:
      accuracy: 0.4675
      date: 2021-04-17_09-33-17
      done: false
      experiment_id: ff92523354ac40d1ba16343aa9c712d3
      hostname: devfair017
      iterations_since_restore: 3
      loss: 1.4908176047563553
      node_ip: 100.97.17.135
      pid: 2642092
      should_checkpoint: true
      time_since_restore: 127.23849749565125
      time_this_iter_s: 40.055989027023315
      time_total_s: 127.23849749565125
      timestamp: 1618677197
      timesteps_since_restore: 0
      training_iteration: 3
      trial_id: 504de_00009
  
    == Status ==
    Memory usage on this node: 35.5/251.8 GiB
    Using AsyncHyperBand: num_stopped=3
    Bracket: Iter 8.000: None | Iter 4.000: -1.623851123201847 | Iter 2.000: -1.8422595864295959 | Iter 1.000: -2.060588000881672
    Resources requested: 14/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (7 RUNNING, 3 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.3869  |     0.5122 |                    4 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.8608  |     0.2646 |                    4 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.18735 |     0.5767 |                    4 |
    | DEFAULT_504de_00005 | RUNNING    | 100.97.17.135:2642086 |            2 |   32 |   16 | 0.00280664  | 1.8247  |     0.3313 |                    1 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.53918 |     0.4203 |                    1 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31773 |     0.1013 |                    4 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.49082 |     0.4675 |                    3 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642085)[0m [5,  4000] loss: 0.540
    [2m[36m(pid=2642086)[0m [2, 16000] loss: 0.217
    [2m[36m(pid=2642083)[0m [2, 16000] loss: 0.178
    Result for DEFAULT_504de_00007:
      accuracy: 0.1004
      date: 2021-04-17_09-33-20
      done: false
      experiment_id: 7adc784bd3e140a9a1bd3db81ac1e8f2
      hostname: devfair017
      iterations_since_restore: 5
      loss: 2.307810732460022
      node_ip: 100.97.17.135
      pid: 2642090
      should_checkpoint: true
      time_since_restore: 129.67567610740662
      time_this_iter_s: 24.951937437057495
      time_total_s: 129.67567610740662
      timestamp: 1618677200
      timesteps_since_restore: 0
      training_iteration: 5
      trial_id: 504de_00007
  
    Result for DEFAULT_504de_00002:
      accuracy: 0.3433
      date: 2021-04-17_09-33-20
      done: false
      experiment_id: da3434fb89c5469e87ceb3aad2c37783
      hostname: devfair017
      iterations_since_restore: 5
      loss: 1.7404713534355163
      node_ip: 100.97.17.135
      pid: 2642088
      should_checkpoint: true
      time_since_restore: 130.25700688362122
      time_this_iter_s: 25.00885272026062
      time_total_s: 130.25700688362122
      timestamp: 1618677200
      timesteps_since_restore: 0
      training_iteration: 5
      trial_id: 504de_00002
  
    Result for DEFAULT_504de_00001:
      accuracy: 0.5481
      date: 2021-04-17_09-33-23
      done: false
      experiment_id: c44a3113e5784b0f83f85e6a34ce7cc5
      hostname: devfair017
      iterations_since_restore: 5
      loss: 1.3170403035879135
      node_ip: 100.97.17.135
      pid: 2642082
      should_checkpoint: true
      time_since_restore: 132.77333307266235
      time_this_iter_s: 25.27439284324646
      time_total_s: 132.77333307266235
      timestamp: 1618677203
      timesteps_since_restore: 0
      training_iteration: 5
      trial_id: 504de_00001
  
    == Status ==
    Memory usage on this node: 35.5/251.8 GiB
    Using AsyncHyperBand: num_stopped=3
    Bracket: Iter 8.000: None | Iter 4.000: -1.623851123201847 | Iter 2.000: -1.8422595864295959 | Iter 1.000: -2.060588000881672
    Resources requested: 14/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (7 RUNNING, 3 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.31704 |     0.5481 |                    5 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.74047 |     0.3433 |                    5 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.18735 |     0.5767 |                    4 |
    | DEFAULT_504de_00005 | RUNNING    | 100.97.17.135:2642086 |            2 |   32 |   16 | 0.00280664  | 1.8247  |     0.3313 |                    1 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.53918 |     0.4203 |                    1 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.30781 |     0.1004 |                    5 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.49082 |     0.4675 |                    3 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642092)[0m [4,  2000] loss: 1.330
    [2m[36m(pid=2642086)[0m [2, 18000] loss: 0.191
    [2m[36m(pid=2642083)[0m [2, 18000] loss: 0.160
    Result for DEFAULT_504de_00004:
      accuracy: 0.5933
      date: 2021-04-17_09-33-26
      done: false
      experiment_id: 5212d8943b3340e09d2704ffe1649c65
      hostname: devfair017
      iterations_since_restore: 5
      loss: 1.168437176299095
      node_ip: 100.97.17.135
      pid: 2642085
      should_checkpoint: true
      time_since_restore: 136.2553436756134
      time_this_iter_s: 25.959039449691772
      time_total_s: 136.2553436756134
      timestamp: 1618677206
      timesteps_since_restore: 0
      training_iteration: 5
      trial_id: 504de_00004
  
    [2m[36m(pid=2642090)[0m [6,  2000] loss: 2.311
    [2m[36m(pid=2642088)[0m [6,  2000] loss: 1.730
    [2m[36m(pid=2642092)[0m [4,  4000] loss: 0.669
    [2m[36m(pid=2642082)[0m [6,  2000] loss: 1.206
    [2m[36m(pid=2642086)[0m [2, 20000] loss: 0.172
    [2m[36m(pid=2642083)[0m [2, 20000] loss: 0.140
    [2m[36m(pid=2642085)[0m [6,  2000] loss: 1.004
    [2m[36m(pid=2642090)[0m [6,  4000] loss: 1.155
    [2m[36m(pid=2642088)[0m [6,  4000] loss: 0.851
    [2m[36m(pid=2642092)[0m [4,  6000] loss: 0.447
    [2m[36m(pid=2642082)[0m [6,  4000] loss: 0.603
    Result for DEFAULT_504de_00005:
      accuracy: 0.323
      date: 2021-04-17_09-33-41
      done: true
      experiment_id: adee13b093d94872921b71add63a3059
      hostname: devfair017
      iterations_since_restore: 2
      loss: 1.8562269247911871
      node_ip: 100.97.17.135
      pid: 2642086
      should_checkpoint: true
      time_since_restore: 150.68497896194458
      time_this_iter_s: 71.98714303970337
      time_total_s: 150.68497896194458
      timestamp: 1618677221
      timesteps_since_restore: 0
      training_iteration: 2
      trial_id: 504de_00005
  
    == Status ==
    Memory usage on this node: 35.6/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: None | Iter 4.000: -1.623851123201847 | Iter 2.000: -1.8562269247911871 | Iter 1.000: -2.060588000881672
    Resources requested: 14/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (7 RUNNING, 3 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.31704 |     0.5481 |                    5 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.74047 |     0.3433 |                    5 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.16844 |     0.5933 |                    5 |
    | DEFAULT_504de_00005 | RUNNING    | 100.97.17.135:2642086 |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.53918 |     0.4203 |                    1 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.30781 |     0.1004 |                    5 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.49082 |     0.4675 |                    3 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00006:
      accuracy: 0.5022
      date: 2021-04-17_09-33-41
      done: false
      experiment_id: 0ca109a83c054e1a9950179524f91481
      hostname: devfair017
      iterations_since_restore: 2
      loss: 1.3839141950763763
      node_ip: 100.97.17.135
      pid: 2642083
      should_checkpoint: true
      time_since_restore: 151.15905809402466
      time_this_iter_s: 72.08936882019043
      time_total_s: 151.15905809402466
      timestamp: 1618677221
      timesteps_since_restore: 0
      training_iteration: 2
      trial_id: 504de_00006
  
    [2m[36m(pid=2642085)[0m [6,  4000] loss: 0.516
    Result for DEFAULT_504de_00007:
      accuracy: 0.0967
      date: 2021-04-17_09-33-45
      done: false
      experiment_id: 7adc784bd3e140a9a1bd3db81ac1e8f2
      hostname: devfair017
      iterations_since_restore: 6
      loss: 2.3145491857528686
      node_ip: 100.97.17.135
      pid: 2642090
      should_checkpoint: true
      time_since_restore: 154.59878849983215
      time_this_iter_s: 24.923112392425537
      time_total_s: 154.59878849983215
      timestamp: 1618677225
      timesteps_since_restore: 0
      training_iteration: 6
      trial_id: 504de_00007
  
    Result for DEFAULT_504de_00002:
      accuracy: 0.3607
      date: 2021-04-17_09-33-45
      done: false
      experiment_id: da3434fb89c5469e87ceb3aad2c37783
      hostname: devfair017
      iterations_since_restore: 6
      loss: 1.6651987537384034
      node_ip: 100.97.17.135
      pid: 2642088
      should_checkpoint: true
      time_since_restore: 155.22220468521118
      time_this_iter_s: 24.965197801589966
      time_total_s: 155.22220468521118
      timestamp: 1618677225
      timesteps_since_restore: 0
      training_iteration: 6
      trial_id: 504de_00002
  
    [2m[36m(pid=2642092)[0m [4,  8000] loss: 0.341
    [2m[36m(pid=2642083)[0m [3,  2000] loss: 1.356
    Result for DEFAULT_504de_00001:
      accuracy: 0.5663
      date: 2021-04-17_09-33-48
      done: false
      experiment_id: c44a3113e5784b0f83f85e6a34ce7cc5
      hostname: devfair017
      iterations_since_restore: 6
      loss: 1.2807460764408112
      node_ip: 100.97.17.135
      pid: 2642082
      should_checkpoint: true
      time_since_restore: 158.1217906475067
      time_this_iter_s: 25.34845757484436
      time_total_s: 158.1217906475067
      timestamp: 1618677228
      timesteps_since_restore: 0
      training_iteration: 6
      trial_id: 504de_00001
  
    == Status ==
    Memory usage on this node: 35.1/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: None | Iter 4.000: -1.623851123201847 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.28075 |     0.5663 |                    6 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.6652  |     0.3607 |                    6 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.16844 |     0.5933 |                    5 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.38391 |     0.5022 |                    2 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31455 |     0.0967 |                    6 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.49082 |     0.4675 |                    3 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00004:
      accuracy: 0.5842
      date: 2021-04-17_09-33-52
      done: false
      experiment_id: 5212d8943b3340e09d2704ffe1649c65
      hostname: devfair017
      iterations_since_restore: 6
      loss: 1.2163499180674553
      node_ip: 100.97.17.135
      pid: 2642085
      should_checkpoint: true
      time_since_restore: 162.26022005081177
      time_this_iter_s: 26.004876375198364
      time_total_s: 162.26022005081177
      timestamp: 1618677232
      timesteps_since_restore: 0
      training_iteration: 6
      trial_id: 504de_00004
  
    [2m[36m(pid=2642092)[0m [4, 10000] loss: 0.272
    [2m[36m(pid=2642090)[0m [7,  2000] loss: 2.311
    [2m[36m(pid=2642083)[0m [3,  4000] loss: 0.694
    [2m[36m(pid=2642088)[0m [7,  2000] loss: 1.669
    [2m[36m(pid=2642082)[0m [7,  2000] loss: 1.161
    Result for DEFAULT_504de_00009:
      accuracy: 0.4988
      date: 2021-04-17_09-33-57
      done: false
      experiment_id: ff92523354ac40d1ba16343aa9c712d3
      hostname: devfair017
      iterations_since_restore: 4
      loss: 1.3719080618858337
      node_ip: 100.97.17.135
      pid: 2642092
      should_checkpoint: true
      time_since_restore: 167.40938258171082
      time_this_iter_s: 40.17088508605957
      time_total_s: 167.40938258171082
      timestamp: 1618677237
      timesteps_since_restore: 0
      training_iteration: 4
      trial_id: 504de_00009
  
    == Status ==
    Memory usage on this node: 35.2/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: None | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.28075 |     0.5663 |                    6 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.6652  |     0.3607 |                    6 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.21635 |     0.5842 |                    6 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.38391 |     0.5022 |                    2 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31455 |     0.0967 |                    6 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.37191 |     0.4988 |                    4 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [3,  6000] loss: 0.462
    [2m[36m(pid=2642085)[0m [7,  2000] loss: 0.949
    [2m[36m(pid=2642090)[0m [7,  4000] loss: 1.156
    [2m[36m(pid=2642088)[0m [7,  4000] loss: 0.814
    [2m[36m(pid=2642092)[0m [5,  2000] loss: 1.312
    [2m[36m(pid=2642082)[0m [7,  4000] loss: 0.599
    [2m[36m(pid=2642083)[0m [3,  8000] loss: 0.337
    Result for DEFAULT_504de_00007:
      accuracy: 0.1013
      date: 2021-04-17_09-34-09
      done: false
      experiment_id: 7adc784bd3e140a9a1bd3db81ac1e8f2
      hostname: devfair017
      iterations_since_restore: 7
      loss: 2.306818337249756
      node_ip: 100.97.17.135
      pid: 2642090
      should_checkpoint: true
      time_since_restore: 179.38910245895386
      time_this_iter_s: 24.790313959121704
      time_total_s: 179.38910245895386
      timestamp: 1618677249
      timesteps_since_restore: 0
      training_iteration: 7
      trial_id: 504de_00007
  
    == Status ==
    Memory usage on this node: 35.2/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: None | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.28075 |     0.5663 |                    6 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.6652  |     0.3607 |                    6 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.21635 |     0.5842 |                    6 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.38391 |     0.5022 |                    2 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.30682 |     0.1013 |                    7 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.37191 |     0.4988 |                    4 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00002:
      accuracy: 0.3918
      date: 2021-04-17_09-34-10
      done: false
      experiment_id: da3434fb89c5469e87ceb3aad2c37783
      hostname: devfair017
      iterations_since_restore: 7
      loss: 1.5997081347465516
      node_ip: 100.97.17.135
      pid: 2642088
      should_checkpoint: true
      time_since_restore: 179.9145884513855
      time_this_iter_s: 24.692383766174316
      time_total_s: 179.9145884513855
      timestamp: 1618677250
      timesteps_since_restore: 0
      training_iteration: 7
      trial_id: 504de_00002
  
    [2m[36m(pid=2642085)[0m [7,  4000] loss: 0.491
    [2m[36m(pid=2642092)[0m [5,  4000] loss: 0.648
    [2m[36m(pid=2642083)[0m [3, 10000] loss: 0.271
    Result for DEFAULT_504de_00001:
      accuracy: 0.554
      date: 2021-04-17_09-34-13
      done: false
      experiment_id: c44a3113e5784b0f83f85e6a34ce7cc5
      hostname: devfair017
      iterations_since_restore: 7
      loss: 1.3296898621797562
      node_ip: 100.97.17.135
      pid: 2642082
      should_checkpoint: true
      time_since_restore: 183.2204167842865
      time_this_iter_s: 25.098626136779785
      time_total_s: 183.2204167842865
      timestamp: 1618677253
      timesteps_since_restore: 0
      training_iteration: 7
      trial_id: 504de_00001
  
    Result for DEFAULT_504de_00004:
      accuracy: 0.6024
      date: 2021-04-17_09-34-18
      done: false
      experiment_id: 5212d8943b3340e09d2704ffe1649c65
      hostname: devfair017
      iterations_since_restore: 7
      loss: 1.1763928109288215
      node_ip: 100.97.17.135
      pid: 2642085
      should_checkpoint: true
      time_since_restore: 187.97410106658936
      time_this_iter_s: 25.713881015777588
      time_total_s: 187.97410106658936
      timestamp: 1618677258
      timesteps_since_restore: 0
      training_iteration: 7
      trial_id: 504de_00004
  
    == Status ==
    Memory usage on this node: 35.3/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: None | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.32969 |     0.554  |                    7 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.59971 |     0.3918 |                    7 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.17639 |     0.6024 |                    7 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.38391 |     0.5022 |                    2 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.30682 |     0.1013 |                    7 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.37191 |     0.4988 |                    4 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642090)[0m [8,  2000] loss: 2.311
    [2m[36m(pid=2642092)[0m [5,  6000] loss: 0.439
    [2m[36m(pid=2642088)[0m [8,  2000] loss: 1.597
    [2m[36m(pid=2642083)[0m [3, 12000] loss: 0.227
    [2m[36m(pid=2642082)[0m [8,  2000] loss: 1.131
    [2m[36m(pid=2642092)[0m [5,  8000] loss: 0.330
    [2m[36m(pid=2642083)[0m [3, 14000] loss: 0.196
    [2m[36m(pid=2642090)[0m [8,  4000] loss: 1.156
    [2m[36m(pid=2642085)[0m [8,  2000] loss: 0.895
    [2m[36m(pid=2642088)[0m [8,  4000] loss: 0.783
    [2m[36m(pid=2642082)[0m [8,  4000] loss: 0.584
    [2m[36m(pid=2642083)[0m [3, 16000] loss: 0.170
    [2m[36m(pid=2642092)[0m [5, 10000] loss: 0.267
    Result for DEFAULT_504de_00007:
      accuracy: 0.1013
      date: 2021-04-17_09-34-34
      done: false
      experiment_id: 7adc784bd3e140a9a1bd3db81ac1e8f2
      hostname: devfair017
      iterations_since_restore: 8
      loss: 2.3133473627090453
      node_ip: 100.97.17.135
      pid: 2642090
      should_checkpoint: true
      time_since_restore: 204.00537133216858
      time_this_iter_s: 24.61626887321472
      time_total_s: 204.00537133216858
      timestamp: 1618677274
      timesteps_since_restore: 0
      training_iteration: 8
      trial_id: 504de_00007
  
    == Status ==
    Memory usage on this node: 35.3/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: -2.3133473627090453 | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.32969 |     0.554  |                    7 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.59971 |     0.3918 |                    7 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.17639 |     0.6024 |                    7 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.38391 |     0.5022 |                    2 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31335 |     0.1013 |                    8 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.37191 |     0.4988 |                    4 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00002:
      accuracy: 0.4198
      date: 2021-04-17_09-34-35
      done: false
      experiment_id: da3434fb89c5469e87ceb3aad2c37783
      hostname: devfair017
      iterations_since_restore: 8
      loss: 1.5469764202594758
      node_ip: 100.97.17.135
      pid: 2642088
      should_checkpoint: true
      time_since_restore: 204.48266220092773
      time_this_iter_s: 24.568073749542236
      time_total_s: 204.48266220092773
      timestamp: 1618677275
      timesteps_since_restore: 0
      training_iteration: 8
      trial_id: 504de_00002
  
    [2m[36m(pid=2642085)[0m [8,  4000] loss: 0.470
    Result for DEFAULT_504de_00009:
      accuracy: 0.5376
      date: 2021-04-17_09-34-37
      done: false
      experiment_id: ff92523354ac40d1ba16343aa9c712d3
      hostname: devfair017
      iterations_since_restore: 5
      loss: 1.3220929484426975
      node_ip: 100.97.17.135
      pid: 2642092
      should_checkpoint: true
      time_since_restore: 207.12867736816406
      time_this_iter_s: 39.71929478645325
      time_total_s: 207.12867736816406
      timestamp: 1618677277
      timesteps_since_restore: 0
      training_iteration: 5
      trial_id: 504de_00009
  
    [2m[36m(pid=2642083)[0m [3, 18000] loss: 0.154
    Result for DEFAULT_504de_00001:
      accuracy: 0.5699
      date: 2021-04-17_09-34-38
      done: false
      experiment_id: c44a3113e5784b0f83f85e6a34ce7cc5
      hostname: devfair017
      iterations_since_restore: 8
      loss: 1.3171062487125396
      node_ip: 100.97.17.135
      pid: 2642082
      should_checkpoint: true
      time_since_restore: 208.21616077423096
      time_this_iter_s: 24.995743989944458
      time_total_s: 208.21616077423096
      timestamp: 1618677278
      timesteps_since_restore: 0
      training_iteration: 8
      trial_id: 504de_00001
  
    [2m[36m(pid=2642090)[0m [9,  2000] loss: 2.311
    [2m[36m(pid=2642088)[0m [9,  2000] loss: 1.548
    Result for DEFAULT_504de_00004:
      accuracy: 0.5846
      date: 2021-04-17_09-34-44
      done: false
      experiment_id: 5212d8943b3340e09d2704ffe1649c65
      hostname: devfair017
      iterations_since_restore: 8
      loss: 1.2665043505907059
      node_ip: 100.97.17.135
      pid: 2642085
      should_checkpoint: true
      time_since_restore: 213.63753175735474
      time_this_iter_s: 25.66343069076538
      time_total_s: 213.63753175735474
      timestamp: 1618677284
      timesteps_since_restore: 0
      training_iteration: 8
      trial_id: 504de_00004
  
    == Status ==
    Memory usage on this node: 35.4/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: -1.4320413344860077 | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.31711 |     0.5699 |                    8 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.54698 |     0.4198 |                    8 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.2665  |     0.5846 |                    8 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.38391 |     0.5022 |                    2 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31335 |     0.1013 |                    8 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.32209 |     0.5376 |                    5 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [3, 20000] loss: 0.134
    [2m[36m(pid=2642092)[0m [6,  2000] loss: 1.278
    [2m[36m(pid=2642082)[0m [9,  2000] loss: 1.126
    [2m[36m(pid=2642092)[0m [6,  4000] loss: 0.645
    [2m[36m(pid=2642090)[0m [9,  4000] loss: 1.156
    [2m[36m(pid=2642088)[0m [9,  4000] loss: 0.759
    [2m[36m(pid=2642085)[0m [9,  2000] loss: 0.853
    Result for DEFAULT_504de_00006:
      accuracy: 0.5362
      date: 2021-04-17_09-34-53
      done: false
      experiment_id: 0ca109a83c054e1a9950179524f91481
      hostname: devfair017
      iterations_since_restore: 3
      loss: 1.3073199604480528
      node_ip: 100.97.17.135
      pid: 2642083
      should_checkpoint: true
      time_since_restore: 222.9391143321991
      time_this_iter_s: 71.78005623817444
      time_total_s: 222.9391143321991
      timestamp: 1618677293
      timesteps_since_restore: 0
      training_iteration: 3
      trial_id: 504de_00006
  
    == Status ==
    Memory usage on this node: 35.4/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: -1.4320413344860077 | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.31711 |     0.5699 |                    8 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.54698 |     0.4198 |                    8 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.2665  |     0.5846 |                    8 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.30732 |     0.5362 |                    3 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31335 |     0.1013 |                    8 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.32209 |     0.5376 |                    5 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642082)[0m [9,  4000] loss: 0.591
    [2m[36m(pid=2642092)[0m [6,  6000] loss: 0.433
    Result for DEFAULT_504de_00007:
      accuracy: 0.1013
      date: 2021-04-17_09-34-59
      done: false
      experiment_id: 7adc784bd3e140a9a1bd3db81ac1e8f2
      hostname: devfair017
      iterations_since_restore: 9
      loss: 2.3030718154907226
      node_ip: 100.97.17.135
      pid: 2642090
      should_checkpoint: true
      time_since_restore: 228.59104919433594
      time_this_iter_s: 24.58567786216736
      time_total_s: 228.59104919433594
      timestamp: 1618677299
      timesteps_since_restore: 0
      training_iteration: 9
      trial_id: 504de_00007
  
    == Status ==
    Memory usage on this node: 35.4/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: -1.4320413344860077 | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.31711 |     0.5699 |                    8 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.54698 |     0.4198 |                    8 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.2665  |     0.5846 |                    8 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.30732 |     0.5362 |                    3 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.30307 |     0.1013 |                    9 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.32209 |     0.5376 |                    5 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00002:
      accuracy: 0.4334
      date: 2021-04-17_09-34-59
      done: false
      experiment_id: da3434fb89c5469e87ceb3aad2c37783
      hostname: devfair017
      iterations_since_restore: 9
      loss: 1.5119693070411682
      node_ip: 100.97.17.135
      pid: 2642088
      should_checkpoint: true
      time_since_restore: 229.1223587989807
      time_this_iter_s: 24.63969659805298
      time_total_s: 229.1223587989807
      timestamp: 1618677299
      timesteps_since_restore: 0
      training_iteration: 9
      trial_id: 504de_00002
  
    [2m[36m(pid=2642083)[0m [4,  2000] loss: 1.308
    [2m[36m(pid=2642085)[0m [9,  4000] loss: 0.452
    Result for DEFAULT_504de_00001:
      accuracy: 0.5648
      date: 2021-04-17_09-35-03
      done: false
      experiment_id: c44a3113e5784b0f83f85e6a34ce7cc5
      hostname: devfair017
      iterations_since_restore: 9
      loss: 1.3049541541337968
      node_ip: 100.97.17.135
      pid: 2642082
      should_checkpoint: true
      time_since_restore: 233.16528511047363
      time_this_iter_s: 24.949124336242676
      time_total_s: 233.16528511047363
      timestamp: 1618677303
      timesteps_since_restore: 0
      training_iteration: 9
      trial_id: 504de_00001
  
    [2m[36m(pid=2642092)[0m [6,  8000] loss: 0.324
    [2m[36m(pid=2642083)[0m [4,  4000] loss: 0.652
    [2m[36m(pid=2642090)[0m [10,  2000] loss: 2.312
    [2m[36m(pid=2642088)[0m [10,  2000] loss: 1.484
    Result for DEFAULT_504de_00004:
      accuracy: 0.5976
      date: 2021-04-17_09-35-09
      done: false
      experiment_id: 5212d8943b3340e09d2704ffe1649c65
      hostname: devfair017
      iterations_since_restore: 9
      loss: 1.212444976013899
      node_ip: 100.97.17.135
      pid: 2642085
      should_checkpoint: true
      time_since_restore: 239.15384984016418
      time_this_iter_s: 25.51631808280945
      time_total_s: 239.15384984016418
      timestamp: 1618677309
      timesteps_since_restore: 0
      training_iteration: 9
      trial_id: 504de_00004
  
    == Status ==
    Memory usage on this node: 35.5/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: -1.4320413344860077 | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.30495 |     0.5648 |                    9 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.51197 |     0.4334 |                    9 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.21244 |     0.5976 |                    9 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.30732 |     0.5362 |                    3 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.30307 |     0.1013 |                    9 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.32209 |     0.5376 |                    5 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [4,  6000] loss: 0.445
    [2m[36m(pid=2642092)[0m [6, 10000] loss: 0.260
    [2m[36m(pid=2642082)[0m [10,  2000] loss: 1.105
    [2m[36m(pid=2642090)[0m [10,  4000] loss: 1.156
    [2m[36m(pid=2642088)[0m [10,  4000] loss: 0.735
    Result for DEFAULT_504de_00009:
      accuracy: 0.5542
      date: 2021-04-17_09-35-17
      done: false
      experiment_id: ff92523354ac40d1ba16343aa9c712d3
      hostname: devfair017
      iterations_since_restore: 6
      loss: 1.2967052514910697
      node_ip: 100.97.17.135
      pid: 2642092
      should_checkpoint: true
      time_since_restore: 246.8367636203766
      time_this_iter_s: 39.708086252212524
      time_total_s: 246.8367636203766
      timestamp: 1618677317
      timesteps_since_restore: 0
      training_iteration: 6
      trial_id: 504de_00009
  
    == Status ==
    Memory usage on this node: 35.5/251.8 GiB
    Using AsyncHyperBand: num_stopped=4
    Bracket: Iter 8.000: -1.4320413344860077 | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.30495 |     0.5648 |                    9 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.51197 |     0.4334 |                    9 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.21244 |     0.5976 |                    9 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.30732 |     0.5362 |                    3 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.30307 |     0.1013 |                    9 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.29671 |     0.5542 |                    6 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [4,  8000] loss: 0.328
    [2m[36m(pid=2642085)[0m [10,  2000] loss: 0.817
    [2m[36m(pid=2642082)[0m [10,  4000] loss: 0.574
    Result for DEFAULT_504de_00007:
      accuracy: 0.1046
      date: 2021-04-17_09-35-23
      done: true
      experiment_id: 7adc784bd3e140a9a1bd3db81ac1e8f2
      hostname: devfair017
      iterations_since_restore: 10
      loss: 2.319181784629822
      node_ip: 100.97.17.135
      pid: 2642090
      should_checkpoint: true
      time_since_restore: 253.28666400909424
      time_this_iter_s: 24.6956148147583
      time_total_s: 253.28666400909424
      timestamp: 1618677323
      timesteps_since_restore: 0
      training_iteration: 10
      trial_id: 504de_00007
  
    == Status ==
    Memory usage on this node: 35.5/251.8 GiB
    Using AsyncHyperBand: num_stopped=5
    Bracket: Iter 8.000: -1.4320413344860077 | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 12/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (6 RUNNING, 4 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00001 | RUNNING    | 100.97.17.135:2642082 |            8 |  128 |   32 | 0.00531987  | 1.30495 |     0.5648 |                    9 |
    | DEFAULT_504de_00002 | RUNNING    | 100.97.17.135:2642088 |            8 |   32 |    4 | 0.000197384 | 1.51197 |     0.4334 |                    9 |
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.21244 |     0.5976 |                    9 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.30732 |     0.5362 |                    3 |
    | DEFAULT_504de_00007 | RUNNING    | 100.97.17.135:2642090 |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.29671 |     0.5542 |                    6 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00002:
      accuracy: 0.4447
      date: 2021-04-17_09-35-24
      done: true
      experiment_id: da3434fb89c5469e87ceb3aad2c37783
      hostname: devfair017
      iterations_since_restore: 10
      loss: 1.4755442904472351
      node_ip: 100.97.17.135
      pid: 2642088
      should_checkpoint: true
      time_since_restore: 253.83177256584167
      time_this_iter_s: 24.709413766860962
      time_total_s: 253.83177256584167
      timestamp: 1618677324
      timesteps_since_restore: 0
      training_iteration: 10
      trial_id: 504de_00002
  
    [2m[36m(pid=2642092)[0m [7,  2000] loss: 1.252
    [2m[36m(pid=2642083)[0m [4, 10000] loss: 0.267
    [2m[36m(pid=2642085)[0m [10,  4000] loss: 0.431
    Result for DEFAULT_504de_00001:
      accuracy: 0.5389
      date: 2021-04-17_09-35-28
      done: true
      experiment_id: c44a3113e5784b0f83f85e6a34ce7cc5
      hostname: devfair017
      iterations_since_restore: 10
      loss: 1.4110066090345383
      node_ip: 100.97.17.135
      pid: 2642082
      should_checkpoint: true
      time_since_restore: 258.0773503780365
      time_this_iter_s: 24.912065267562866
      time_total_s: 258.0773503780365
      timestamp: 1618677328
      timesteps_since_restore: 0
      training_iteration: 10
      trial_id: 504de_00001
  
    [2m[36m(pid=2642083)[0m [4, 12000] loss: 0.212
    [2m[36m(pid=2642092)[0m [7,  4000] loss: 0.648
    Result for DEFAULT_504de_00004:
      accuracy: 0.6041
      date: 2021-04-17_09-35-34
      done: true
      experiment_id: 5212d8943b3340e09d2704ffe1649c65
      hostname: devfair017
      iterations_since_restore: 10
      loss: 1.207703362518549
      node_ip: 100.97.17.135
      pid: 2642085
      should_checkpoint: true
      time_since_restore: 264.42176270484924
      time_this_iter_s: 25.26791286468506
      time_total_s: 264.42176270484924
      timestamp: 1618677334
      timesteps_since_restore: 0
      training_iteration: 10
      trial_id: 504de_00004
  
    == Status ==
    Memory usage on this node: 33.5/251.8 GiB
    Using AsyncHyperBand: num_stopped=8
    Bracket: Iter 8.000: -1.4320413344860077 | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 6/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (3 RUNNING, 7 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00004 | RUNNING    | 100.97.17.135:2642085 |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.30732 |     0.5362 |                    3 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.29671 |     0.5542 |                    6 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [4, 14000] loss: 0.191
    [2m[36m(pid=2642092)[0m [7,  6000] loss: 0.431
    [2m[36m(pid=2642083)[0m [4, 16000] loss: 0.171
    [2m[36m(pid=2642092)[0m [7,  8000] loss: 0.319
    [2m[36m(pid=2642083)[0m [4, 18000] loss: 0.147
    [2m[36m(pid=2642092)[0m [7, 10000] loss: 0.260
    [2m[36m(pid=2642083)[0m [4, 20000] loss: 0.135
    Result for DEFAULT_504de_00009:
      accuracy: 0.5357
      date: 2021-04-17_09-35-56
      done: false
      experiment_id: ff92523354ac40d1ba16343aa9c712d3
      hostname: devfair017
      iterations_since_restore: 7
      loss: 1.3737160887021571
      node_ip: 100.97.17.135
      pid: 2642092
      should_checkpoint: true
      time_since_restore: 285.7331967353821
      time_this_iter_s: 38.89643311500549
      time_total_s: 285.7331967353821
      timestamp: 1618677356
      timesteps_since_restore: 0
      training_iteration: 7
      trial_id: 504de_00009
  
    == Status ==
    Memory usage on this node: 32.9/251.8 GiB
    Using AsyncHyperBand: num_stopped=8
    Bracket: Iter 8.000: -1.4320413344860077 | Iter 4.000: -1.3869041404962539 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 4/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (2 RUNNING, 8 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.30732 |     0.5362 |                    3 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.37372 |     0.5357 |                    7 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |                       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00006:
      accuracy: 0.5437
      date: 2021-04-17_09-36-02
      done: false
      experiment_id: 0ca109a83c054e1a9950179524f91481
      hostname: devfair017
      iterations_since_restore: 4
      loss: 1.3141421076660975
      node_ip: 100.97.17.135
      pid: 2642083
      should_checkpoint: true
      time_since_restore: 292.4437527656555
      time_this_iter_s: 69.50463843345642
      time_total_s: 292.4437527656555
      timestamp: 1618677362
      timesteps_since_restore: 0
      training_iteration: 4
      trial_id: 504de_00006
  
    == Status ==
    Memory usage on this node: 32.9/251.8 GiB
    Using AsyncHyperBand: num_stopped=8
    Bracket: Iter 8.000: -1.4320413344860077 | Iter 4.000: -1.379406101191044 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 4/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (2 RUNNING, 8 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.31414 |     0.5437 |                    4 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.37372 |     0.5357 |                    7 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |                       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642092)[0m [8,  2000] loss: 1.250
    [2m[36m(pid=2642083)[0m [5,  2000] loss: 1.273
    [2m[36m(pid=2642092)[0m [8,  4000] loss: 0.632
    [2m[36m(pid=2642083)[0m [5,  4000] loss: 0.651
    [2m[36m(pid=2642092)[0m [8,  6000] loss: 0.430
    [2m[36m(pid=2642083)[0m [5,  6000] loss: 0.426
    [2m[36m(pid=2642092)[0m [8,  8000] loss: 0.323
    [2m[36m(pid=2642083)[0m [5,  8000] loss: 0.321
    [2m[36m(pid=2642092)[0m [8, 10000] loss: 0.255
    [2m[36m(pid=2642083)[0m [5, 10000] loss: 0.263
    Result for DEFAULT_504de_00009:
      accuracy: 0.5438
      date: 2021-04-17_09-36-34
      done: false
      experiment_id: ff92523354ac40d1ba16343aa9c712d3
      hostname: devfair017
      iterations_since_restore: 8
      loss: 1.325555314488709
      node_ip: 100.97.17.135
      pid: 2642092
      should_checkpoint: true
      time_since_restore: 323.9386696815491
      time_this_iter_s: 38.20547294616699
      time_total_s: 323.9386696815491
      timestamp: 1618677394
      timesteps_since_restore: 0
      training_iteration: 8
      trial_id: 504de_00009
  
    == Status ==
    Memory usage on this node: 32.9/251.8 GiB
    Using AsyncHyperBand: num_stopped=8
    Bracket: Iter 8.000: -1.325555314488709 | Iter 4.000: -1.379406101191044 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 4/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (2 RUNNING, 8 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.31414 |     0.5437 |                    4 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.32556 |     0.5438 |                    8 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |                       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [5, 12000] loss: 0.213
    [2m[36m(pid=2642092)[0m [9,  2000] loss: 1.254
    [2m[36m(pid=2642083)[0m [5, 14000] loss: 0.191
    [2m[36m(pid=2642092)[0m [9,  4000] loss: 0.616
    [2m[36m(pid=2642083)[0m [5, 16000] loss: 0.165
    [2m[36m(pid=2642092)[0m [9,  6000] loss: 0.422
    [2m[36m(pid=2642083)[0m [5, 18000] loss: 0.146
    [2m[36m(pid=2642092)[0m [9,  8000] loss: 0.314
    [2m[36m(pid=2642083)[0m [5, 20000] loss: 0.134
    [2m[36m(pid=2642092)[0m [9, 10000] loss: 0.255
    Result for DEFAULT_504de_00006:
      accuracy: 0.5264
      date: 2021-04-17_09-37-11
      done: false
      experiment_id: 0ca109a83c054e1a9950179524f91481
      hostname: devfair017
      iterations_since_restore: 5
      loss: 1.3598943022679304
      node_ip: 100.97.17.135
      pid: 2642083
      should_checkpoint: true
      time_since_restore: 361.0877778530121
      time_this_iter_s: 68.64402508735657
      time_total_s: 361.0877778530121
      timestamp: 1618677431
      timesteps_since_restore: 0
      training_iteration: 5
      trial_id: 504de_00006
  
    == Status ==
    Memory usage on this node: 32.9/251.8 GiB
    Using AsyncHyperBand: num_stopped=8
    Bracket: Iter 8.000: -1.325555314488709 | Iter 4.000: -1.379406101191044 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 4/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (2 RUNNING, 8 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.35989 |     0.5264 |                    5 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.32556 |     0.5438 |                    8 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |                       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    Result for DEFAULT_504de_00009:
      accuracy: 0.5094
      date: 2021-04-17_09-37-12
      done: false
      experiment_id: ff92523354ac40d1ba16343aa9c712d3
      hostname: devfair017
      iterations_since_restore: 9
      loss: 1.4525348477542401
      node_ip: 100.97.17.135
      pid: 2642092
      should_checkpoint: true
      time_since_restore: 362.2753360271454
      time_this_iter_s: 38.33666634559631
      time_total_s: 362.2753360271454
      timestamp: 1618677432
      timesteps_since_restore: 0
      training_iteration: 9
      trial_id: 504de_00009
  
    [2m[36m(pid=2642083)[0m [6,  2000] loss: 1.283
    [2m[36m(pid=2642092)[0m [10,  2000] loss: 1.253
    [2m[36m(pid=2642083)[0m [6,  4000] loss: 0.650
    [2m[36m(pid=2642092)[0m [10,  4000] loss: 0.636
    [2m[36m(pid=2642083)[0m [6,  6000] loss: 0.415
    [2m[36m(pid=2642092)[0m [10,  6000] loss: 0.413
    [2m[36m(pid=2642083)[0m [6,  8000] loss: 0.323
    [2m[36m(pid=2642092)[0m [10,  8000] loss: 0.312
    [2m[36m(pid=2642083)[0m [6, 10000] loss: 0.257
    [2m[36m(pid=2642092)[0m [10, 10000] loss: 0.253
    [2m[36m(pid=2642083)[0m [6, 12000] loss: 0.215
    Result for DEFAULT_504de_00009:
      accuracy: 0.5338
      date: 2021-04-17_09-37-51
      done: true
      experiment_id: ff92523354ac40d1ba16343aa9c712d3
      hostname: devfair017
      iterations_since_restore: 10
      loss: 1.3585052899122239
      node_ip: 100.97.17.135
      pid: 2642092
      should_checkpoint: true
      time_since_restore: 400.8685886859894
      time_this_iter_s: 38.593252658843994
      time_total_s: 400.8685886859894
      timestamp: 1618677471
      timesteps_since_restore: 0
      training_iteration: 10
      trial_id: 504de_00009
  
    == Status ==
    Memory usage on this node: 33.0/251.8 GiB
    Using AsyncHyperBand: num_stopped=9
    Bracket: Iter 8.000: -1.325555314488709 | Iter 4.000: -1.379406101191044 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 4/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (2 RUNNING, 8 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.35989 |     0.5264 |                    5 |
    | DEFAULT_504de_00009 | RUNNING    | 100.97.17.135:2642092 |            4 |   16 |   16 | 0.00211468  | 1.35851 |     0.5338 |                   10 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |                       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [6, 14000] loss: 0.185
    [2m[36m(pid=2642083)[0m [6, 16000] loss: 0.160
    [2m[36m(pid=2642083)[0m [6, 18000] loss: 0.141
    [2m[36m(pid=2642083)[0m [6, 20000] loss: 0.131
    Result for DEFAULT_504de_00006:
      accuracy: 0.5234
      date: 2021-04-17_09-38-20
      done: false
      experiment_id: 0ca109a83c054e1a9950179524f91481
      hostname: devfair017
      iterations_since_restore: 6
      loss: 1.397710119656165
      node_ip: 100.97.17.135
      pid: 2642083
      should_checkpoint: true
      time_since_restore: 429.93652629852295
      time_this_iter_s: 68.84874844551086
      time_total_s: 429.93652629852295
      timestamp: 1618677500
      timesteps_since_restore: 0
      training_iteration: 6
      trial_id: 504de_00006
  
    == Status ==
    Memory usage on this node: 32.3/251.8 GiB
    Using AsyncHyperBand: num_stopped=9
    Bracket: Iter 8.000: -1.325555314488709 | Iter 4.000: -1.379406101191044 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 2/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (1 RUNNING, 9 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.39771 |     0.5234 |                    6 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |                       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    | DEFAULT_504de_00009 | TERMINATED |                       |            4 |   16 |   16 | 0.00211468  | 1.35851 |     0.5338 |                   10 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [7,  2000] loss: 1.301
    [2m[36m(pid=2642083)[0m [7,  4000] loss: 0.624
    [2m[36m(pid=2642083)[0m [7,  6000] loss: 0.410
    [2m[36m(pid=2642083)[0m [7,  8000] loss: 0.325
    [2m[36m(pid=2642083)[0m [7, 10000] loss: 0.254
    [2m[36m(pid=2642083)[0m [7, 12000] loss: 0.214
    [2m[36m(pid=2642083)[0m [7, 14000] loss: 0.180
    [2m[36m(pid=2642083)[0m [7, 16000] loss: 0.162
    [2m[36m(pid=2642083)[0m [7, 18000] loss: 0.146
    [2m[36m(pid=2642083)[0m [7, 20000] loss: 0.127
    Result for DEFAULT_504de_00006:
      accuracy: 0.5423
      date: 2021-04-17_09-39-28
      done: false
      experiment_id: 0ca109a83c054e1a9950179524f91481
      hostname: devfair017
      iterations_since_restore: 7
      loss: 1.3143013595831348
      node_ip: 100.97.17.135
      pid: 2642083
      should_checkpoint: true
      time_since_restore: 498.42230701446533
      time_this_iter_s: 68.48578071594238
      time_total_s: 498.42230701446533
      timestamp: 1618677568
      timesteps_since_restore: 0
      training_iteration: 7
      trial_id: 504de_00006
  
    == Status ==
    Memory usage on this node: 32.3/251.8 GiB
    Using AsyncHyperBand: num_stopped=9
    Bracket: Iter 8.000: -1.325555314488709 | Iter 4.000: -1.379406101191044 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 2/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (1 RUNNING, 9 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.3143  |     0.5423 |                    7 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |                       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    | DEFAULT_504de_00009 | TERMINATED |                       |            4 |   16 |   16 | 0.00211468  | 1.35851 |     0.5338 |                   10 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [8,  2000] loss: 1.251
    [2m[36m(pid=2642083)[0m [8,  4000] loss: 0.630
    [2m[36m(pid=2642083)[0m [8,  6000] loss: 0.419
    [2m[36m(pid=2642083)[0m [8,  8000] loss: 0.331
    [2m[36m(pid=2642083)[0m [8, 10000] loss: 0.253
    Result for DEFAULT_504de_00006:
    [2m[36m(pid=2642083)[0m [8, 12000] loss: 0.213
      accuracy: 0.5619
      date: 2021-04-17_09-40-37
      done: false
      experiment_id: 0ca109a83c054e1a9950179524f91481
      hostname: devfair017
      iterations_since_restore: 8
      loss: 1.2921368273621425
      node_ip: 100.97.17.135
      pid: 2642083
      should_checkpoint: true
      time_since_restore: 567.0916693210602
      time_this_iter_s: 68.66936230659485
      time_total_s: 567.0916693210602
      timestamp: 1618677637
      timesteps_since_restore: 0
      training_iteration: 8
      trial_id: 504de_00006
      [2m[36m(pid=2642083)[0m [8, 14000] loss: 0.181
    [2m[36m(pid=2642083)[0m [8, 16000] loss: 0.157
    [2m[36m(pid=2642083)[0m [8, 18000] loss: 0.143
    [2m[36m(pid=2642083)[0m [8, 20000] loss: 0.127

    == Status ==
    Memory usage on this node: 32.4/251.8 GiB
    Using AsyncHyperBand: num_stopped=9
    Bracket: Iter 8.000: -1.3213307816006243 | Iter 4.000: -1.379406101191044 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 2/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (1 RUNNING, 9 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.29214 |     0.5619 |                    8 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |                       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    | DEFAULT_504de_00009 | TERMINATED |                       |            4 |   16 |   16 | 0.00211468  | 1.35851 |     0.5338 |                   10 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [9,  2000] loss: 1.246
    [2m[36m(pid=2642083)[0m [9,  4000] loss: 0.624
    [2m[36m(pid=2642083)[0m [9,  6000] loss: 0.420
    [2m[36m(pid=2642083)[0m [9,  8000] loss: 0.304
    [2m[36m(pid=2642083)[0m [9, 10000] loss: 0.249
    [2m[36m(pid=2642083)[0m [9, 12000] loss: 0.206
    [2m[36m(pid=2642083)[0m [9, 14000] loss: 0.178
    [2m[36m(pid=2642083)[0m [9, 16000] loss: 0.160
    [2m[36m(pid=2642083)[0m [9, 18000] loss: 0.143
    [2m[36m(pid=2642083)[0m [9, 20000] loss: 0.128
    Result for DEFAULT_504de_00006:
      accuracy: 0.5375
      date: 2021-04-17_10-04-46
      done: false
      experiment_id: 0ca109a83c054e1a9950179524f91481
      hostname: devfair017
      iterations_since_restore: 9
      loss: 1.3217835801563924
      node_ip: 100.97.17.135
      pid: 2642083
      should_checkpoint: true
      time_since_restore: 2016.358303785324
      time_this_iter_s: 1449.266634464264
      time_total_s: 2016.358303785324
      timestamp: 1618679086
      timesteps_since_restore: 0
      training_iteration: 9
      trial_id: 504de_00006
  
    == Status ==
    Memory usage on this node: 32.4/251.8 GiB
    Using AsyncHyperBand: num_stopped=9
    Bracket: Iter 8.000: -1.3213307816006243 | Iter 4.000: -1.379406101191044 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 2/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (1 RUNNING, 9 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.32178 |     0.5375 |                    9 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |                       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    | DEFAULT_504de_00009 | TERMINATED |                       |            4 |   16 |   16 | 0.00211468  | 1.35851 |     0.5338 |                   10 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    [2m[36m(pid=2642083)[0m [10,  2000] loss: 1.214
    [2m[36m(pid=2642083)[0m [10,  4000] loss: 0.620
    [2m[36m(pid=2642083)[0m [10,  6000] loss: 0.408
    [2m[36m(pid=2642083)[0m [10,  8000] loss: 0.312
    [2m[36m(pid=2642083)[0m [10, 10000] loss: 0.251
    [2m[36m(pid=2642083)[0m [10, 12000] loss: 0.207
    [2m[36m(pid=2642083)[0m [10, 14000] loss: 0.184
    [2m[36m(pid=2642083)[0m [10, 16000] loss: 0.153
    [2m[36m(pid=2642083)[0m [10, 18000] loss: 0.141
    [2m[36m(pid=2642083)[0m [10, 20000] loss: 0.126
    Result for DEFAULT_504de_00006:
      accuracy: 0.5296
      date: 2021-04-17_10-05-55
      done: true
      experiment_id: 0ca109a83c054e1a9950179524f91481
      hostname: devfair017
      iterations_since_restore: 10
      loss: 1.383230909213191
      node_ip: 100.97.17.135
      pid: 2642083
      should_checkpoint: true
      time_since_restore: 2085.15460729599
      time_this_iter_s: 68.7963035106659
      time_total_s: 2085.15460729599
      timestamp: 1618679155
      timesteps_since_restore: 0
      training_iteration: 10
      trial_id: 504de_00006
  
    == Status ==
    Memory usage on this node: 32.4/251.8 GiB
    Using AsyncHyperBand: num_stopped=10
    Bracket: Iter 8.000: -1.3213307816006243 | Iter 4.000: -1.379406101191044 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 2/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (1 RUNNING, 9 TERMINATED)
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc                   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00006 | RUNNING    | 100.97.17.135:2642083 |            2 |   16 |  256 | 0.00111242  | 1.38323 |     0.5296 |                   10 |
    | DEFAULT_504de_00000 | TERMINATED |                       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |                       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |                       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |                       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |                       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |                       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00007 | TERMINATED |                       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |                       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    | DEFAULT_504de_00009 | TERMINATED |                       |            4 |   16 |   16 | 0.00211468  | 1.35851 |     0.5338 |                   10 |
    +---------------------+------------+-----------------------+--------------+------+------+-------------+---------+------------+----------------------+


    == Status ==
    Memory usage on this node: 32.3/251.8 GiB
    Using AsyncHyperBand: num_stopped=10
    Bracket: Iter 8.000: -1.3213307816006243 | Iter 4.000: -1.379406101191044 | Iter 2.000: -1.649679903979227 | Iter 1.000: -2.060588000881672
    Resources requested: 0/24 CPUs, 0/2 GPUs, 0.0/146.73 GiB heap, 0.0/46.14 GiB objects (0/1.0 accelerator_type:GP100)
    Result logdir: /private/home/howardhuang/ray_results/DEFAULT_2021-04-17_09-31-08
    Number of trials: 10/10 (10 TERMINATED)
    +---------------------+------------+-------+--------------+------+------+-------------+---------+------------+----------------------+
    | Trial name          | status     | loc   |   batch_size |   l1 |   l2 |          lr |    loss |   accuracy |   training_iteration |
    |---------------------+------------+-------+--------------+------+------+-------------+---------+------------+----------------------|
    | DEFAULT_504de_00000 | TERMINATED |       |            4 |  128 |    8 | 0.0233167   | 2.31313 |     0.1002 |                    1 |
    | DEFAULT_504de_00001 | TERMINATED |       |            8 |  128 |   32 | 0.00531987  | 1.41101 |     0.5389 |                   10 |
    | DEFAULT_504de_00002 | TERMINATED |       |            8 |   32 |    4 | 0.000197384 | 1.47554 |     0.4447 |                   10 |
    | DEFAULT_504de_00003 | TERMINATED |       |            8 |    8 |  256 | 0.0184795   | 2.30448 |     0.1006 |                    2 |
    | DEFAULT_504de_00004 | TERMINATED |       |            8 |  128 |  128 | 0.00233347  | 1.2077  |     0.6041 |                   10 |
    | DEFAULT_504de_00005 | TERMINATED |       |            2 |   32 |   16 | 0.00280664  | 1.85623 |     0.323  |                    2 |
    | DEFAULT_504de_00006 | TERMINATED |       |            2 |   16 |  256 | 0.00111242  | 1.38323 |     0.5296 |                   10 |
    | DEFAULT_504de_00007 | TERMINATED |       |            8 |    8 |   16 | 0.0301274   | 2.31918 |     0.1046 |                   10 |
    | DEFAULT_504de_00008 | TERMINATED |       |            8 |    8 |  128 | 0.0388454   | 2.32749 |     0.1002 |                    1 |
    | DEFAULT_504de_00009 | TERMINATED |       |            4 |   16 |   16 | 0.00211468  | 1.35851 |     0.5338 |                   10 |
    +---------------------+------------+-------+--------------+------+------+-------------+---------+------------+----------------------+


    Best trial config: {'l1': 128, 'l2': 128, 'lr': 0.0023334723524965056, 'batch_size': 8}
    Best trial final validation loss: 1.207703362518549
    Best trial final validation accuracy: 0.6041
    Files already downloaded and verified
    Files already downloaded and verified
    Best trial test set accuracy: 0.6001


If you run the code, an example output could look like this:

.. code-block::

    Number of trials: 10 (10 TERMINATED)
    +-----+------+------+-------------+--------------+---------+------------+--------------------+
    | ... |   l1 |   l2 |          lr |   batch_size |    loss |   accuracy | training_iteration |
    |-----+------+------+-------------+--------------+---------+------------+--------------------|
    | ... |   64 |    4 | 0.00011629  |            2 | 1.87273 |     0.244  |                  2 |
    | ... |   32 |   64 | 0.000339763 |            8 | 1.23603 |     0.567  |                  8 |
    | ... |    8 |   16 | 0.00276249  |           16 | 1.1815  |     0.5836 |                 10 |
    | ... |    4 |   64 | 0.000648721 |            4 | 1.31131 |     0.5224 |                  8 |
    | ... |   32 |   16 | 0.000340753 |            8 | 1.26454 |     0.5444 |                  8 |
    | ... |    8 |    4 | 0.000699775 |            8 | 1.99594 |     0.1983 |                  2 |
    | ... |  256 |    8 | 0.0839654   |           16 | 2.3119  |     0.0993 |                  1 |
    | ... |   16 |  128 | 0.0758154   |           16 | 2.33575 |     0.1327 |                  1 |
    | ... |   16 |    8 | 0.0763312   |           16 | 2.31129 |     0.1042 |                  4 |
    | ... |  128 |   16 | 0.000124903 |            4 | 2.26917 |     0.1945 |                  1 |
    +-----+------+------+-------------+--------------+---------+------------+--------------------+


    Best trial config: {'l1': 8, 'l2': 16, 'lr': 0.00276249, 'batch_size': 16, 'data_dir': '...'}
    Best trial final validation loss: 1.181501
    Best trial final validation accuracy: 0.5836
    Best trial test set accuracy: 0.5806

Most trials have been stopped early in order to avoid wasting resources.
The best performing trial achieved a validation accuracy of about 58%, which could
be confirmed on the test set.

So that's it! You can now tune the parameters of your PyTorch models.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 34 minutes  59.760 seconds)


.. _sphx_glr_download_beginner_hyperparameter_tuning_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: hyperparameter_tuning_tutorial.py <hyperparameter_tuning_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: hyperparameter_tuning_tutorial.ipynb <hyperparameter_tuning_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
