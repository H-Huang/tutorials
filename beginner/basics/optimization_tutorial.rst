.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_beginner_basics_optimization_tutorial.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_beginner_basics_optimization_tutorial.py:


`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ || 
`Tensors <tensorqs_tutorial.html>`_ || 
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
**Optimization** ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Optimizing Model Parameters
===========================

Now that we have a model and data it's time to train, validate and test our model by optimizing it's parameters on 
our data. Training a model is an iterative process; in each iteration (called an *epoch*) the model makes a guess about the output, calculates 
the error in its guess (*loss*), collects the derivatives of the error with respect to its parameters (as we saw in 
the `previous section  <autograd_tutorial.html>`_), and **optimizes** these parameters using gradient descent. For a more 
detailed walkthrough of this process, check out this video on `backpropagation from 3Blue1Brown <https://www.youtube.com/watch?v=tIeHLnjs5U8>`__.

Pre-requisite Code 
-----------------
We load the code from the previous sections on `Datasets & DataLoaders <data_tutorial.html>`_ 
and `Build Model  <buildmodel_tutorial.html>`_.

.. code-block:: default


    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor, Lambda

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

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

    model = NeuralNetwork()








Hyperparameters 
-----------------

Hyperparameters are adjustable parameters that let you control the model optimization process. 
Different hyperparameter values can impact model training and convergence rates 
(`read more <https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html>`__ about hyperparameter tuning)

We define the following hyperparameters for training:
 - **Number of Epochs** - the number times to iterate over the dataset
 - **Batch Size** - the number of data samples seen by the model in each epoch
 - **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.



.. code-block:: default


    learning_rate = 1e-3
    batch_size = 64
    epochs = 5









Optimization Loop
-----------------

Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each 
iteration of the optimization loop is called an **epoch**. 

Each epoch consists of two main parts:
 - **The Train Loop** - iterate over the training dataset and try to converge to optimal parameters.
 - **The Validation/Test Loop** - iterate over the test dataset to check if model performance is improving.

Let's briefly familiarize ourselves with some of the concepts used in the training loop. Jump ahead to 
see the :ref:`full-impl-label` of the optimization loop.

Loss Function
~~~~~~~~~~~~~~~~~

When presented with some training data, our untrained network is likely not to give the correct 
answer. **Loss function** measures the degree of dissimilarity of obtained result to the target value, 
and it is the loss function that we want to minimize during training. To calculate the loss we make a 
prediction using the inputs of our given data sample and compare it against the true data label value.

Common loss functions include `nn.MSELoss <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss>`_ (Mean Square Error) for regression tasks, and 
`nn.NLLLoss <https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss>`_ (Negative Log Likelihood) for classification. 
`nn.CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss>`_ combines ``nn.LogSoftmax`` and ``nn.NLLLoss``.

We pass our model's output logits to ``nn.CrossEntropyLoss``, which will normalize the logits and compute the prediction error.


.. code-block:: default


    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()









Optimizer
~~~~~~~~~~~~~~~~~

Optimization is the process of adjusting model parameters to reduce model error in each training step. **Optimization algorithms** define how this process is performed (in this example we use Stochastic Gradient Descent).
All optimization logic is encapsulated in  the ``optimizer`` object. Here, we use the SGD optimizer; additionally, there are many `different optimizers <https://pytorch.org/docs/stable/optim.html>`_ 
available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models and data.

We initialize the optimizer by registering the model's parameters that need to be trained, and passing in the learning rate hyperparameter.


.. code-block:: default


    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)







Inside the training loop, optimization happens in three steps:
 * Call ``optimizer.zero_grad()`` to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
 * Backpropagate the prediction loss with a call to ``loss.backwards()``. PyTorch deposits the gradients of the loss w.r.t. each parameter. 
 * Once we have our gradients, we call ``optimizer.step()`` to adjust the parameters by the gradients collected in the backward pass.  

.. _full-impl-label:

Full Implementation
-----------------------
We define ``train_loop`` that loops over our optimization code, and ``test_loop`` that 
evaluates the model's performance against our test data.


.. code-block:: default


    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):        
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
        
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")








We initialize the loss function and optimizer, and pass it to ``train_loop`` and ``test_loop``.
Feel free to increase the number of epochs to track the model's improving performance.


.. code-block:: default


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")







.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Epoch 1
    -------------------------------
    loss: 2.305578  [    0/60000]
    loss: 2.299938  [ 6400/60000]
    loss: 2.295990  [12800/60000]
    loss: 2.292274  [19200/60000]
    loss: 2.276649  [25600/60000]
    loss: 2.267849  [32000/60000]
    loss: 2.266357  [38400/60000]
    loss: 2.258584  [44800/60000]
    loss: 2.262501  [51200/60000]
    loss: 2.223772  [57600/60000]
    Test Error: 
     Accuracy: 30.1%, Avg loss: 0.035235 

    Epoch 2
    -------------------------------
    loss: 2.246869  [    0/60000]
    loss: 2.252240  [ 6400/60000]
    loss: 2.228384  [12800/60000]
    loss: 2.238943  [19200/60000]
    loss: 2.201072  [25600/60000]
    loss: 2.166404  [32000/60000]
    loss: 2.189749  [38400/60000]
    loss: 2.162585  [44800/60000]
    loss: 2.190232  [51200/60000]
    loss: 2.097346  [57600/60000]
    Test Error: 
     Accuracy: 29.6%, Avg loss: 0.033790 

    Epoch 3
    -------------------------------
    loss: 2.161041  [    0/60000]
    loss: 2.176069  [ 6400/60000]
    loss: 2.123772  [12800/60000]
    loss: 2.155610  [19200/60000]
    loss: 2.073602  [25600/60000]
    loss: 2.011877  [32000/60000]
    loss: 2.069308  [38400/60000]
    loss: 2.017369  [44800/60000]
    loss: 2.086261  [51200/60000]
    loss: 1.917799  [57600/60000]
    Test Error: 
     Accuracy: 30.5%, Avg loss: 0.031807 

    Epoch 4
    -------------------------------
    loss: 2.040473  [    0/60000]
    loss: 2.073391  [ 6400/60000]
    loss: 1.996865  [12800/60000]
    loss: 2.053465  [19200/60000]
    loss: 1.923691  [25600/60000]
    loss: 1.856643  [32000/60000]
    loss: 1.931949  [38400/60000]
    loss: 1.876680  [44800/60000]
    loss: 1.969269  [51200/60000]
    loss: 1.747970  [57600/60000]
    Test Error: 
     Accuracy: 32.2%, Avg loss: 0.029963 

    Epoch 5
    -------------------------------
    loss: 1.917498  [    0/60000]
    loss: 1.979733  [ 6400/60000]
    loss: 1.889974  [12800/60000]
    loss: 1.959330  [19200/60000]
    loss: 1.797620  [25600/60000]
    loss: 1.744750  [32000/60000]
    loss: 1.809715  [38400/60000]
    loss: 1.767351  [44800/60000]
    loss: 1.860058  [51200/60000]
    loss: 1.616440  [57600/60000]
    Test Error: 
     Accuracy: 34.0%, Avg loss: 0.028470 

    Epoch 6
    -------------------------------
    loss: 1.810390  [    0/60000]
    loss: 1.903203  [ 6400/60000]
    loss: 1.802062  [12800/60000]
    loss: 1.885040  [19200/60000]
    loss: 1.700277  [25600/60000]
    loss: 1.661405  [32000/60000]
    loss: 1.720592  [38400/60000]
    loss: 1.687159  [44800/60000]
    loss: 1.780153  [51200/60000]
    loss: 1.526374  [57600/60000]
    Test Error: 
     Accuracy: 35.3%, Avg loss: 0.027360 

    Epoch 7
    -------------------------------
    loss: 1.730018  [    0/60000]
    loss: 1.841127  [ 6400/60000]
    loss: 1.733185  [12800/60000]
    loss: 1.830371  [19200/60000]
    loss: 1.627783  [25600/60000]
    loss: 1.601272  [32000/60000]
    loss: 1.656791  [38400/60000]
    loss: 1.626793  [44800/60000]
    loss: 1.722562  [51200/60000]
    loss: 1.464559  [57600/60000]
    Test Error: 
     Accuracy: 36.3%, Avg loss: 0.026520 

    Epoch 8
    -------------------------------
    loss: 1.669566  [    0/60000]
    loss: 1.791628  [ 6400/60000]
    loss: 1.677427  [12800/60000]
    loss: 1.788723  [19200/60000]
    loss: 1.575772  [25600/60000]
    loss: 1.556721  [32000/60000]
    loss: 1.611821  [38400/60000]
    loss: 1.580487  [44800/60000]
    loss: 1.680362  [51200/60000]
    loss: 1.420193  [57600/60000]
    Test Error: 
     Accuracy: 37.3%, Avg loss: 0.025866 

    Epoch 9
    -------------------------------
    loss: 1.624670  [    0/60000]
    loss: 1.752705  [ 6400/60000]
    loss: 1.630588  [12800/60000]
    loss: 1.755201  [19200/60000]
    loss: 1.537656  [25600/60000]
    loss: 1.522115  [32000/60000]
    loss: 1.578289  [38400/60000]
    loss: 1.542597  [44800/60000]
    loss: 1.646913  [51200/60000]
    loss: 1.387020  [57600/60000]
    Test Error: 
     Accuracy: 38.4%, Avg loss: 0.025342 

    Epoch 10
    -------------------------------
    loss: 1.587753  [    0/60000]
    loss: 1.720477  [ 6400/60000]
    loss: 1.590909  [12800/60000]
    loss: 1.727927  [19200/60000]
    loss: 1.508027  [25600/60000]
    loss: 1.494340  [32000/60000]
    loss: 1.550869  [38400/60000]
    loss: 1.511539  [44800/60000]
    loss: 1.618494  [51200/60000]
    loss: 1.361107  [57600/60000]
    Test Error: 
     Accuracy: 39.3%, Avg loss: 0.024908 

    Done!


Further Reading
-----------------------
- `Loss Functions <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
- `torch.optim <https://pytorch.org/docs/stable/optim.html>`_
- `Warmstart Training a Model <https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html>`_



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 9 minutes  26.008 seconds)


.. _sphx_glr_download_beginner_basics_optimization_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: optimization_tutorial.py <optimization_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: optimization_tutorial.ipynb <optimization_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
