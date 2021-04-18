.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_beginner_basics_quickstart_tutorial.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_beginner_basics_quickstart_tutorial.py:


`Learn the Basics <intro.html>`_ ||
**Quickstart** || 
`Tensors <tensorqs_tutorial.html>`_ || 
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Quickstart
===================
This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.

Working with data
-----------------
PyTorch has two `primitives to work with data <https://pytorch.org/docs/stable/data.html>`_: 
``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``.
``Dataset`` stores the samples and their corresponding labels, and ``DataLoader`` wraps an iterable around
the ``Dataset``.

.. code-block:: default


    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor, Lambda, Compose
    import matplotlib.pyplot as plt







PyTorch offers domain-specific libraries such as `TorchText <https://pytorch.org/text/stable/index.html>`_, 
`TorchVision <https://pytorch.org/vision/stable/index.html>`_, and `TorchAudio <https://pytorch.org/audio/stable/index.html>`_, 
all of which include datasets. For this tutorial, we  will be using a TorchVision dataset.

The ``torchvision.datasets`` module contains ``Dataset`` objects for many real-world vision data like 
CIFAR, COCO (`full list here <https://pytorch.org/docs/stable/torchvision/datasets.html>`_). In this tutorial, we
use the FashionMNIST dataset. Every TorchVision ``Dataset`` includes two arguments: ``transform`` and
``target_transform`` to modify the samples and labels respectively.


.. code-block:: default


    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )







We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element 
in the dataloader iterable will return a batch of 64 features and labels.


.. code-block:: default


    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
    Shape of y:  torch.Size([64]) torch.int64


Read more about `loading data in PyTorch <data_tutorial.html>`_.


--------------


Creating Models
------------------
To define a neural network in PyTorch, we create a class that inherits 
from `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_. We define the layers of the network
in the ``__init__`` function and specify how data will pass through the network in the ``forward`` function. To accelerate 
operations in the neural network, we move it to the GPU if available.


.. code-block:: default


    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
                nn.ReLU()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    print(model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Using cuda device
    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
        (5): ReLU()
      )
    )


Read more about `building neural networks in PyTorch <buildmodel_tutorial.html>`_.


--------------


Optimizing the Model Parameters
----------------------------------------
To train a model, we need a `loss function <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
and an `optimizer <https://pytorch.org/docs/stable/optim.html>`_. 


.. code-block:: default


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)








In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and 
backpropagates the prediction error to adjust the model's parameters. 


.. code-block:: default


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
        
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
        
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")







We also check the model's performance against the test dataset to ensure it is learning.


.. code-block:: default


    def test(dataloader, model):
        size = len(dataloader.dataset)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")







The training process is conducted over several iterations (*epochs*). During each epoch, the model learns 
parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the
accuracy increase and the loss decrease with every epoch.


.. code-block:: default


    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)
    print("Done!")





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Epoch 1
    -------------------------------
    loss: 2.305943  [    0/60000]
    loss: 2.294084  [ 6400/60000]
    loss: 2.288995  [12800/60000]
    loss: 2.278657  [19200/60000]
    loss: 2.269294  [25600/60000]
    loss: 2.272419  [32000/60000]
    loss: 2.262859  [38400/60000]
    loss: 2.268046  [44800/60000]
    loss: 2.248616  [51200/60000]
    loss: 2.211506  [57600/60000]
    Test Error: 
     Accuracy: 45.1%, Avg loss: 0.034993 

    Epoch 2
    -------------------------------
    loss: 2.258904  [    0/60000]
    loss: 2.240235  [ 6400/60000]
    loss: 2.234893  [12800/60000]
    loss: 2.204126  [19200/60000]
    loss: 2.188091  [25600/60000]
    loss: 2.214937  [32000/60000]
    loss: 2.181956  [38400/60000]
    loss: 2.205982  [44800/60000]
    loss: 2.173014  [51200/60000]
    loss: 2.091019  [57600/60000]
    Test Error: 
     Accuracy: 45.9%, Avg loss: 0.033497 

    Epoch 3
    -------------------------------
    loss: 2.201818  [    0/60000]
    loss: 2.164263  [ 6400/60000]
    loss: 2.160885  [12800/60000]
    loss: 2.093126  [19200/60000]
    loss: 2.071939  [25600/60000]
    loss: 2.130737  [32000/60000]
    loss: 2.053868  [38400/60000]
    loss: 2.099510  [44800/60000]
    loss: 2.029690  [51200/60000]
    loss: 1.918076  [57600/60000]
    Test Error: 
     Accuracy: 45.9%, Avg loss: 0.031072 

    Epoch 4
    -------------------------------
    loss: 2.077690  [    0/60000]
    loss: 2.015167  [ 6400/60000]
    loss: 1.986010  [12800/60000]
    loss: 1.900625  [19200/60000]
    loss: 1.926189  [25600/60000]
    loss: 1.991777  [32000/60000]
    loss: 1.882318  [38400/60000]
    loss: 1.950347  [44800/60000]
    loss: 1.837771  [51200/60000]
    loss: 1.723839  [57600/60000]
    Test Error: 
     Accuracy: 46.1%, Avg loss: 0.028269 

    Epoch 5
    -------------------------------
    loss: 1.916914  [    0/60000]
    loss: 1.844567  [ 6400/60000]
    loss: 1.790600  [12800/60000]
    loss: 1.714792  [19200/60000]
    loss: 1.798896  [25600/60000]
    loss: 1.856342  [32000/60000]
    loss: 1.735227  [38400/60000]
    loss: 1.820971  [44800/60000]
    loss: 1.687082  [51200/60000]
    loss: 1.575495  [57600/60000]
    Test Error: 
     Accuracy: 50.0%, Avg loss: 0.026068 

    Done!


Read more about `Training your model <optimization_tutorial.html>`_.


--------------


Saving Models
-------------
A common way to save a model is to serialize the internal state dictionary (containing the model parameters).


.. code-block:: default


    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")







.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Saved PyTorch Model State to model.pth


Loading Models
----------------------------

The process for loading a model includes re-creating the model structure and loading
the state dictionary into it. 


.. code-block:: default


    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))







This model can now be used to make predictions.


.. code-block:: default


    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

      




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Predicted: "Sneaker", Actual: "Ankle boot"


Read more about `Saving & Loading your model <saveloadrun_tutorial.html>`_.



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  40.661 seconds)


.. _sphx_glr_download_beginner_basics_quickstart_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: quickstart_tutorial.py <quickstart_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: quickstart_tutorial.ipynb <quickstart_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
