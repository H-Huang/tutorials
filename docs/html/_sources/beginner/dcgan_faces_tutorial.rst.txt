.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_beginner_dcgan_faces_tutorial.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_beginner_dcgan_faces_tutorial.py:


DCGAN Tutorial
==============

**Author**: `Nathan Inkawhich <https://github.com/inkawhich>`__
Introduction
------------

This tutorial will give an introduction to DCGANs through an example. We
will train a generative adversarial network (GAN) to generate new
celebrities after showing it pictures of many real celebrities. Most of
the code here is from the dcgan implementation in
`pytorch/examples <https://github.com/pytorch/examples>`__, and this
document will give a thorough explanation of the implementation and shed
light on how and why this model works. But don’t worry, no prior
knowledge of GANs is required, but it may require a first-timer to spend
some time reasoning about what is actually happening under the hood.
Also, for the sake of time it will help to have a GPU, or two. Lets
start from the beginning.

Generative Adversarial Networks
-------------------------------

What is a GAN?
~~~~~~~~~~~~~~

GANs are a framework for teaching a DL model to capture the training
data’s distribution so we can generate new data from that same
distribution. GANs were invented by Ian Goodfellow in 2014 and first
described in the paper `Generative Adversarial
Nets <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`__.
They are made of two distinct models, a *generator* and a
*discriminator*. The job of the generator is to spawn ‘fake’ images that
look like the training images. The job of the discriminator is to look
at an image and output whether or not it is a real training image or a
fake image from the generator. During training, the generator is
constantly trying to outsmart the discriminator by generating better and
better fakes, while the discriminator is working to become a better
detective and correctly classify the real and fake images. The
equilibrium of this game is when the generator is generating perfect
fakes that look as if they came directly from the training data, and the
discriminator is left to always guess at 50% confidence that the
generator output is real or fake.

Now, lets define some notation to be used throughout tutorial starting
with the discriminator. Let :math:`x` be data representing an image.
:math:`D(x)` is the discriminator network which outputs the (scalar)
probability that :math:`x` came from training data rather than the
generator. Here, since we are dealing with images the input to
:math:`D(x)` is an image of CHW size 3x64x64. Intuitively, :math:`D(x)`
should be HIGH when :math:`x` comes from training data and LOW when
:math:`x` comes from the generator. :math:`D(x)` can also be thought of
as a traditional binary classifier.

For the generator’s notation, let :math:`z` be a latent space vector
sampled from a standard normal distribution. :math:`G(z)` represents the
generator function which maps the latent vector :math:`z` to data-space.
The goal of :math:`G` is to estimate the distribution that the training
data comes from (:math:`p_{data}`) so it can generate fake samples from
that estimated distribution (:math:`p_g`).

So, :math:`D(G(z))` is the probability (scalar) that the output of the
generator :math:`G` is a real image. As described in `Goodfellow’s
paper <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`__,
:math:`D` and :math:`G` play a minimax game in which :math:`D` tries to
maximize the probability it correctly classifies reals and fakes
(:math:`logD(x)`), and :math:`G` tries to minimize the probability that
:math:`D` will predict its outputs are fake (:math:`log(1-D(G(x)))`).
From the paper, the GAN loss function is

.. math:: \underset{G}{\text{min}} \underset{D}{\text{max}}V(D,G) = \mathbb{E}_{x\sim p_{data}(x)}\big[logD(x)\big] + \mathbb{E}_{z\sim p_{z}(z)}\big[log(1-D(G(z)))\big]

In theory, the solution to this minimax game is where
:math:`p_g = p_{data}`, and the discriminator guesses randomly if the
inputs are real or fake. However, the convergence theory of GANs is
still being actively researched and in reality models do not always
train to this point.

What is a DCGAN?
~~~~~~~~~~~~~~~~

A DCGAN is a direct extension of the GAN described above, except that it
explicitly uses convolutional and convolutional-transpose layers in the
discriminator and generator, respectively. It was first described by
Radford et. al. in the paper `Unsupervised Representation Learning With
Deep Convolutional Generative Adversarial
Networks <https://arxiv.org/pdf/1511.06434.pdf>`__. The discriminator
is made up of strided
`convolution <https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d>`__
layers, `batch
norm <https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d>`__
layers, and
`LeakyReLU <https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU>`__
activations. The input is a 3x64x64 input image and the output is a
scalar probability that the input is from the real data distribution.
The generator is comprised of
`convolutional-transpose <https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d>`__
layers, batch norm layers, and
`ReLU <https://pytorch.org/docs/stable/nn.html#relu>`__ activations. The
input is a latent vector, :math:`z`, that is drawn from a standard
normal distribution and the output is a 3x64x64 RGB image. The strided
conv-transpose layers allow the latent vector to be transformed into a
volume with the same shape as an image. In the paper, the authors also
give some tips about how to setup the optimizers, how to calculate the
loss functions, and how to initialize the model weights, all of which
will be explained in the coming sections.



.. code-block:: default


    from __future__ import print_function
    #%matplotlib inline
    import argparse
    import os
    import random
    import torch
    import torch.nn as nn
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.optim as optim
    import torch.utils.data
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import torchvision.utils as vutils
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Random Seed:  999


Inputs
------

Let’s define some inputs for the run:

-  **dataroot** - the path to the root of the dataset folder. We will
   talk more about the dataset in the next section
-  **workers** - the number of worker threads for loading the data with
   the DataLoader
-  **batch_size** - the batch size used in training. The DCGAN paper
   uses a batch size of 128
-  **image_size** - the spatial size of the images used for training.
   This implementation defaults to 64x64. If another size is desired,
   the structures of D and G must be changed. See
   `here <https://github.com/pytorch/examples/issues/70>`__ for more
   details
-  **nc** - number of color channels in the input images. For color
   images this is 3
-  **nz** - length of latent vector
-  **ngf** - relates to the depth of feature maps carried through the
   generator
-  **ndf** - sets the depth of feature maps propagated through the
   discriminator
-  **num_epochs** - number of training epochs to run. Training for
   longer will probably lead to better results but will also take much
   longer
-  **lr** - learning rate for training. As described in the DCGAN paper,
   this number should be 0.0002
-  **beta1** - beta1 hyperparameter for Adam optimizers. As described in
   paper, this number should be 0.5
-  **ngpu** - number of GPUs available. If this is 0, code will run in
   CPU mode. If this number is greater than 0 it will run on that number
   of GPUs



.. code-block:: default


    # Root directory for dataset
    dataroot = "data/celeba"

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 5

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1








Data
----

In this tutorial we will use the `Celeb-A Faces
dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__ which can
be downloaded at the linked site, or in `Google
Drive <https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg>`__.
The dataset will download as a file named *img_align_celeba.zip*. Once
downloaded, create a directory named *celeba* and extract the zip file
into that directory. Then, set the *dataroot* input for this notebook to
the *celeba* directory you just created. The resulting directory
structure should be:

::

   /path/to/celeba
       -> img_align_celeba  
           -> 188242.jpg
           -> 173822.jpg
           -> 284702.jpg
           -> 537394.jpg
              ...

This is an important step because we will be using the ImageFolder
dataset class, which requires there to be subdirectories in the
dataset’s root folder. Now, we can create the dataset, create the
dataloader, set the device to run on, and finally visualize some of the
training data.



.. code-block:: default


    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))






.. image:: /beginner/images/sphx_glr_dcgan_faces_tutorial_001.png
    :class: sphx-glr-single-img




Implementation
--------------

With our input parameters set and the dataset prepared, we can now get
into the implementation. We will start with the weight initialization
strategy, then talk about the generator, discriminator, loss functions,
and training loop in detail.

Weight Initialization
~~~~~~~~~~~~~~~~~~~~~

From the DCGAN paper, the authors specify that all model weights shall
be randomly initialized from a Normal distribution with mean=0,
stdev=0.02. The ``weights_init`` function takes an initialized model as
input and reinitializes all convolutional, convolutional-transpose, and
batch normalization layers to meet this criteria. This function is
applied to the models immediately after initialization.



.. code-block:: default


    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)








Generator
~~~~~~~~~

The generator, :math:`G`, is designed to map the latent space vector
(:math:`z`) to data-space. Since our data are images, converting
:math:`z` to data-space means ultimately creating a RGB image with the
same size as the training images (i.e. 3x64x64). In practice, this is
accomplished through a series of strided two dimensional convolutional
transpose layers, each paired with a 2d batch norm layer and a relu
activation. The output of the generator is fed through a tanh function
to return it to the input data range of :math:`[-1,1]`. It is worth
noting the existence of the batch norm functions after the
conv-transpose layers, as this is a critical contribution of the DCGAN
paper. These layers help with the flow of gradients during training. An
image of the generator from the DCGAN paper is shown below.

.. figure:: /_static/img/dcgan_generator.png
   :alt: dcgan_generator

Notice, the how the inputs we set in the input section (*nz*, *ngf*, and
*nc*) influence the generator architecture in code. *nz* is the length
of the z input vector, *ngf* relates to the size of the feature maps
that are propagated through the generator, and *nc* is the number of
channels in the output image (set to 3 for RGB images). Below is the
code for the generator.



.. code-block:: default


    # Generator Code

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)








Now, we can instantiate the generator and apply the ``weights_init``
function. Check out the printed model to see how the generator object is
structured.



.. code-block:: default


    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Generator(
      (main): Sequential(
        (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU(inplace=True)
        (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (11): ReLU(inplace=True)
        (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (13): Tanh()
      )
    )


Discriminator
~~~~~~~~~~~~~

As mentioned, the discriminator, :math:`D`, is a binary classification
network that takes an image as input and outputs a scalar probability
that the input image is real (as opposed to fake). Here, :math:`D` takes
a 3x64x64 input image, processes it through a series of Conv2d,
BatchNorm2d, and LeakyReLU layers, and outputs the final probability
through a Sigmoid activation function. This architecture can be extended
with more layers if necessary for the problem, but there is significance
to the use of the strided convolution, BatchNorm, and LeakyReLUs. The
DCGAN paper mentions it is a good practice to use strided convolution
rather than pooling to downsample because it lets the network learn its
own pooling function. Also batch norm and leaky relu functions promote
healthy gradient flow which is critical for the learning process of both
:math:`G` and :math:`D`.


Discriminator Code


.. code-block:: default


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)








Now, as with the generator, we can create the discriminator, apply the
``weights_init`` function, and print the model’s structure.



.. code-block:: default


    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Discriminator(
      (main): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): LeakyReLU(negative_slope=0.2, inplace=True)
        (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): LeakyReLU(negative_slope=0.2, inplace=True)
        (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (10): LeakyReLU(negative_slope=0.2, inplace=True)
        (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
        (12): Sigmoid()
      )
    )


Loss Functions and Optimizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With :math:`D` and :math:`G` setup, we can specify how they learn
through the loss functions and optimizers. We will use the Binary Cross
Entropy loss
(`BCELoss <https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss>`__)
function which is defined in PyTorch as:

.. math:: \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad l_n = - \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

Notice how this function provides the calculation of both log components
in the objective function (i.e. :math:`log(D(x))` and
:math:`log(1-D(G(z)))`). We can specify what part of the BCE equation to
use with the :math:`y` input. This is accomplished in the training loop
which is coming up soon, but it is important to understand how we can
choose which component we wish to calculate just by changing :math:`y`
(i.e. GT labels).

Next, we define our real label as 1 and the fake label as 0. These
labels will be used when calculating the losses of :math:`D` and
:math:`G`, and this is also the convention used in the original GAN
paper. Finally, we set up two separate optimizers, one for :math:`D` and
one for :math:`G`. As specified in the DCGAN paper, both are Adam
optimizers with learning rate 0.0002 and Beta1 = 0.5. For keeping track
of the generator’s learning progression, we will generate a fixed batch
of latent vectors that are drawn from a Gaussian distribution
(i.e. fixed_noise) . In the training loop, we will periodically input
this fixed_noise into :math:`G`, and over the iterations we will see
images form out of the noise.



.. code-block:: default


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))








Training
~~~~~~~~

Finally, now that we have all of the parts of the GAN framework defined,
we can train it. Be mindful that training GANs is somewhat of an art
form, as incorrect hyperparameter settings lead to mode collapse with
little explanation of what went wrong. Here, we will closely follow
Algorithm 1 from Goodfellow’s paper, while abiding by some of the best
practices shown in `ganhacks <https://github.com/soumith/ganhacks>`__.
Namely, we will “construct different mini-batches for real and fake”
images, and also adjust G’s objective function to maximize
:math:`logD(G(z))`. Training is split up into two main parts. Part 1
updates the Discriminator and Part 2 updates the Generator.

**Part 1 - Train the Discriminator**

Recall, the goal of training the discriminator is to maximize the
probability of correctly classifying a given input as real or fake. In
terms of Goodfellow, we wish to “update the discriminator by ascending
its stochastic gradient”. Practically, we want to maximize
:math:`log(D(x)) + log(1-D(G(z)))`. Due to the separate mini-batch
suggestion from ganhacks, we will calculate this in two steps. First, we
will construct a batch of real samples from the training set, forward
pass through :math:`D`, calculate the loss (:math:`log(D(x))`), then
calculate the gradients in a backward pass. Secondly, we will construct
a batch of fake samples with the current generator, forward pass this
batch through :math:`D`, calculate the loss (:math:`log(1-D(G(z)))`),
and *accumulate* the gradients with a backward pass. Now, with the
gradients accumulated from both the all-real and all-fake batches, we
call a step of the Discriminator’s optimizer.

**Part 2 - Train the Generator**

As stated in the original paper, we want to train the Generator by
minimizing :math:`log(1-D(G(z)))` in an effort to generate better fakes.
As mentioned, this was shown by Goodfellow to not provide sufficient
gradients, especially early in the learning process. As a fix, we
instead wish to maximize :math:`log(D(G(z)))`. In the code we accomplish
this by: classifying the Generator output from Part 1 with the
Discriminator, computing G’s loss *using real labels as GT*, computing
G’s gradients in a backward pass, and finally updating G’s parameters
with an optimizer step. It may seem counter-intuitive to use the real
labels as GT labels for the loss function, but this allows us to use the
:math:`log(x)` part of the BCELoss (rather than the :math:`log(1-x)`
part) which is exactly what we want.

Finally, we will do some statistic reporting and at the end of each
epoch we will push our fixed_noise batch through the generator to
visually track the progress of G’s training. The training statistics
reported are:

-  **Loss_D** - discriminator loss calculated as the sum of losses for
   the all real and all fake batches (:math:`log(D(x)) + log(1 - D(G(z)))`).
-  **Loss_G** - generator loss calculated as :math:`log(D(G(z)))`
-  **D(x)** - the average output (across the batch) of the discriminator
   for the all real batch. This should start close to 1 then
   theoretically converge to 0.5 when G gets better. Think about why
   this is.
-  **D(G(z))** - average discriminator outputs for the all fake batch.
   The first number is before D is updated and the second number is
   after D is updated. These numbers should start near 0 and converge to
   0.5 as G gets better. Think about why this is.

**Note:** This step might take a while, depending on how many epochs you
run and if you removed some data from the dataset.



.. code-block:: default


    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
        
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
        
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            iters += 1






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Starting Training Loop...
    [0/5][0/1583]   Loss_D: 1.7690  Loss_G: 6.5275  D(x): 0.6867    D(G(z)): 0.6682 / 0.0027
    [0/5][50/1583]  Loss_D: 0.0776  Loss_G: 14.2919 D(x): 0.9620    D(G(z)): 0.0000 / 0.0000
    [0/5][100/1583] Loss_D: 0.1257  Loss_G: 7.8988  D(x): 0.9246    D(G(z)): 0.0008 / 0.0008
    [0/5][150/1583] Loss_D: 0.4478  Loss_G: 5.6867  D(x): 0.9109    D(G(z)): 0.2535 / 0.0117
    [0/5][200/1583] Loss_D: 0.9678  Loss_G: 1.8267  D(x): 0.5490    D(G(z)): 0.0567 / 0.2031
    [0/5][250/1583] Loss_D: 0.2672  Loss_G: 5.6664  D(x): 0.8528    D(G(z)): 0.0505 / 0.0091
    [0/5][300/1583] Loss_D: 0.5438  Loss_G: 4.1365  D(x): 0.8397    D(G(z)): 0.2468 / 0.0284
    [0/5][350/1583] Loss_D: 0.3783  Loss_G: 2.5596  D(x): 0.8078    D(G(z)): 0.0983 / 0.1035
    [0/5][400/1583] Loss_D: 0.4671  Loss_G: 3.7897  D(x): 0.8447    D(G(z)): 0.2005 / 0.0372
    [0/5][450/1583] Loss_D: 0.5380  Loss_G: 5.5835  D(x): 0.9533    D(G(z)): 0.3274 / 0.0075
    [0/5][500/1583] Loss_D: 0.5040  Loss_G: 5.6781  D(x): 0.8578    D(G(z)): 0.2543 / 0.0076
    [0/5][550/1583] Loss_D: 1.8485  Loss_G: 2.3802  D(x): 0.2966    D(G(z)): 0.0261 / 0.1465
    [0/5][600/1583] Loss_D: 1.1681  Loss_G: 1.2949  D(x): 0.4505    D(G(z)): 0.0217 / 0.4295
    [0/5][650/1583] Loss_D: 0.4908  Loss_G: 4.7858  D(x): 0.8657    D(G(z)): 0.2164 / 0.0176
    [0/5][700/1583] Loss_D: 0.1724  Loss_G: 5.8330  D(x): 0.9095    D(G(z)): 0.0629 / 0.0062
    [0/5][750/1583] Loss_D: 1.6128  Loss_G: 3.6083  D(x): 0.3519    D(G(z)): 0.0037 / 0.0750
    [0/5][800/1583] Loss_D: 1.2824  Loss_G: 8.7492  D(x): 0.9138    D(G(z)): 0.5953 / 0.0007
    [0/5][850/1583] Loss_D: 0.7225  Loss_G: 6.8074  D(x): 0.8854    D(G(z)): 0.3912 / 0.0027
    [0/5][900/1583] Loss_D: 0.7965  Loss_G: 3.0432  D(x): 0.5838    D(G(z)): 0.0282 / 0.0835
    [0/5][950/1583] Loss_D: 0.6039  Loss_G: 3.4939  D(x): 0.7185    D(G(z)): 0.1442 / 0.0578
    [0/5][1000/1583]        Loss_D: 0.2264  Loss_G: 2.8924  D(x): 0.9243    D(G(z)): 0.1115 / 0.1197
    [0/5][1050/1583]        Loss_D: 0.8711  Loss_G: 7.5156  D(x): 0.9445    D(G(z)): 0.5003 / 0.0011
    [0/5][1100/1583]        Loss_D: 0.5678  Loss_G: 5.8946  D(x): 0.9639    D(G(z)): 0.3548 / 0.0062
    [0/5][1150/1583]        Loss_D: 0.4244  Loss_G: 3.1374  D(x): 0.7692    D(G(z)): 0.0733 / 0.0705
    [0/5][1200/1583]        Loss_D: 0.4716  Loss_G: 4.7156  D(x): 0.8607    D(G(z)): 0.2244 / 0.0176
    [0/5][1250/1583]        Loss_D: 0.5166  Loss_G: 3.5064  D(x): 0.8202    D(G(z)): 0.2224 / 0.0454
    [0/5][1300/1583]        Loss_D: 0.4772  Loss_G: 3.6937  D(x): 0.7690    D(G(z)): 0.1067 / 0.0414
    [0/5][1350/1583]        Loss_D: 0.6079  Loss_G: 3.6760  D(x): 0.7958    D(G(z)): 0.2500 / 0.0389
    [0/5][1400/1583]        Loss_D: 1.1319  Loss_G: 7.7290  D(x): 0.9477    D(G(z)): 0.6017 / 0.0013
    [0/5][1450/1583]        Loss_D: 0.4027  Loss_G: 4.6461  D(x): 0.8781    D(G(z)): 0.1962 / 0.0151
    [0/5][1500/1583]        Loss_D: 0.5065  Loss_G: 2.9459  D(x): 0.7498    D(G(z)): 0.1297 / 0.0793
    [0/5][1550/1583]        Loss_D: 0.6896  Loss_G: 5.8326  D(x): 0.9284    D(G(z)): 0.3900 / 0.0061
    [1/5][0/1583]   Loss_D: 0.6302  Loss_G: 4.8344  D(x): 0.8834    D(G(z)): 0.3374 / 0.0134
    [1/5][50/1583]  Loss_D: 1.4518  Loss_G: 7.1602  D(x): 0.9285    D(G(z)): 0.6721 / 0.0034
    [1/5][100/1583] Loss_D: 1.0506  Loss_G: 1.3801  D(x): 0.4769    D(G(z)): 0.0443 / 0.3410
    [1/5][150/1583] Loss_D: 0.4329  Loss_G: 4.1858  D(x): 0.8715    D(G(z)): 0.2181 / 0.0243
    [1/5][200/1583] Loss_D: 0.3815  Loss_G: 3.5158  D(x): 0.8087    D(G(z)): 0.1142 / 0.0503
    [1/5][250/1583] Loss_D: 1.0663  Loss_G: 1.2463  D(x): 0.4608    D(G(z)): 0.0311 / 0.3664
    [1/5][300/1583] Loss_D: 0.6973  Loss_G: 5.1125  D(x): 0.9212    D(G(z)): 0.3969 / 0.0123
    [1/5][350/1583] Loss_D: 0.4818  Loss_G: 4.2535  D(x): 0.8642    D(G(z)): 0.2456 / 0.0229
    [1/5][400/1583] Loss_D: 0.6969  Loss_G: 5.9982  D(x): 0.9743    D(G(z)): 0.4378 / 0.0050
    [1/5][450/1583] Loss_D: 0.4474  Loss_G: 2.4141  D(x): 0.7229    D(G(z)): 0.0654 / 0.1378
    [1/5][500/1583] Loss_D: 0.4606  Loss_G: 3.1132  D(x): 0.7884    D(G(z)): 0.1577 / 0.0618
    [1/5][550/1583] Loss_D: 0.8408  Loss_G: 5.5430  D(x): 0.9194    D(G(z)): 0.4470 / 0.0096
    [1/5][600/1583] Loss_D: 0.4555  Loss_G: 2.6606  D(x): 0.7843    D(G(z)): 0.1381 / 0.1020
    [1/5][650/1583] Loss_D: 0.3588  Loss_G: 4.0147  D(x): 0.9004    D(G(z)): 0.2021 / 0.0260
    [1/5][700/1583] Loss_D: 0.7281  Loss_G: 4.3235  D(x): 0.6104    D(G(z)): 0.0394 / 0.0310
    [1/5][750/1583] Loss_D: 0.4854  Loss_G: 3.0303  D(x): 0.8123    D(G(z)): 0.1942 / 0.0719
    [1/5][800/1583] Loss_D: 0.5760  Loss_G: 3.6534  D(x): 0.8648    D(G(z)): 0.2987 / 0.0376
    [1/5][850/1583] Loss_D: 1.0518  Loss_G: 5.2000  D(x): 0.9230    D(G(z)): 0.5362 / 0.0150
    [1/5][900/1583] Loss_D: 0.3724  Loss_G: 4.6134  D(x): 0.9477    D(G(z)): 0.2379 / 0.0151
    [1/5][950/1583] Loss_D: 0.3476  Loss_G: 3.2363  D(x): 0.8716    D(G(z)): 0.1685 / 0.0540
    [1/5][1000/1583]        Loss_D: 0.6982  Loss_G: 3.3409  D(x): 0.7953    D(G(z)): 0.3103 / 0.0561
    [1/5][1050/1583]        Loss_D: 0.7232  Loss_G: 1.7973  D(x): 0.5985    D(G(z)): 0.0645 / 0.2221
    [1/5][1100/1583]        Loss_D: 0.4404  Loss_G: 2.5475  D(x): 0.7410    D(G(z)): 0.0953 / 0.1105
    [1/5][1150/1583]        Loss_D: 0.5801  Loss_G: 3.1903  D(x): 0.6391    D(G(z)): 0.0333 / 0.0701
    [1/5][1200/1583]        Loss_D: 0.4820  Loss_G: 2.7538  D(x): 0.7933    D(G(z)): 0.1765 / 0.0907
    [1/5][1250/1583]        Loss_D: 0.4276  Loss_G: 2.8674  D(x): 0.9388    D(G(z)): 0.2658 / 0.0874
    [1/5][1300/1583]        Loss_D: 0.9991  Loss_G: 5.2373  D(x): 0.9202    D(G(z)): 0.5184 / 0.0098
    [1/5][1350/1583]        Loss_D: 0.6211  Loss_G: 3.4172  D(x): 0.8424    D(G(z)): 0.3123 / 0.0488
    [1/5][1400/1583]        Loss_D: 0.6153  Loss_G: 2.9102  D(x): 0.8128    D(G(z)): 0.2942 / 0.0763
    [1/5][1450/1583]        Loss_D: 0.4396  Loss_G: 2.8261  D(x): 0.7766    D(G(z)): 0.1317 / 0.0839
    [1/5][1500/1583]        Loss_D: 0.4386  Loss_G: 3.2041  D(x): 0.8541    D(G(z)): 0.1814 / 0.0665
    [1/5][1550/1583]        Loss_D: 1.1328  Loss_G: 0.6052  D(x): 0.4027    D(G(z)): 0.0355 / 0.5832
    [2/5][0/1583]   Loss_D: 0.4538  Loss_G: 2.5938  D(x): 0.7285    D(G(z)): 0.0923 / 0.0975
    [2/5][50/1583]  Loss_D: 0.6678  Loss_G: 1.4134  D(x): 0.6429    D(G(z)): 0.1235 / 0.2964
    [2/5][100/1583] Loss_D: 0.6191  Loss_G: 3.4348  D(x): 0.8458    D(G(z)): 0.3245 / 0.0432
    [2/5][150/1583] Loss_D: 0.4982  Loss_G: 2.1935  D(x): 0.7056    D(G(z)): 0.0918 / 0.1593
    [2/5][200/1583] Loss_D: 0.9123  Loss_G: 1.6988  D(x): 0.5730    D(G(z)): 0.1505 / 0.2327
    [2/5][250/1583] Loss_D: 0.4998  Loss_G: 2.4760  D(x): 0.7563    D(G(z)): 0.1514 / 0.1118
    [2/5][300/1583] Loss_D: 0.6225  Loss_G: 1.6836  D(x): 0.6556    D(G(z)): 0.1292 / 0.2222
    [2/5][350/1583] Loss_D: 1.0015  Loss_G: 0.4259  D(x): 0.4744    D(G(z)): 0.0612 / 0.7031
    [2/5][400/1583] Loss_D: 0.5611  Loss_G: 2.0536  D(x): 0.7383    D(G(z)): 0.1799 / 0.1640
    [2/5][450/1583] Loss_D: 0.6922  Loss_G: 4.0907  D(x): 0.8986    D(G(z)): 0.3991 / 0.0282
    [2/5][500/1583] Loss_D: 1.1536  Loss_G: 0.7148  D(x): 0.3981    D(G(z)): 0.0436 / 0.5389
    [2/5][550/1583] Loss_D: 0.6708  Loss_G: 2.0489  D(x): 0.6428    D(G(z)): 0.1274 / 0.1772
    [2/5][600/1583] Loss_D: 0.5078  Loss_G: 2.1437  D(x): 0.7661    D(G(z)): 0.1805 / 0.1574
    [2/5][650/1583] Loss_D: 0.8851  Loss_G: 3.4206  D(x): 0.8903    D(G(z)): 0.4792 / 0.0470
    [2/5][700/1583] Loss_D: 0.7734  Loss_G: 3.9334  D(x): 0.9182    D(G(z)): 0.4527 / 0.0291
    [2/5][750/1583] Loss_D: 0.6786  Loss_G: 3.1863  D(x): 0.8914    D(G(z)): 0.3755 / 0.0600
    [2/5][800/1583] Loss_D: 0.5701  Loss_G: 2.1916  D(x): 0.7662    D(G(z)): 0.2260 / 0.1334
    [2/5][850/1583] Loss_D: 1.0810  Loss_G: 1.3848  D(x): 0.4226    D(G(z)): 0.0495 / 0.3128
    [2/5][900/1583] Loss_D: 0.5324  Loss_G: 2.8446  D(x): 0.8228    D(G(z)): 0.2505 / 0.0783
    [2/5][950/1583] Loss_D: 0.6522  Loss_G: 2.9137  D(x): 0.8725    D(G(z)): 0.3562 / 0.0731
    [2/5][1000/1583]        Loss_D: 0.5782  Loss_G: 1.3450  D(x): 0.7242    D(G(z)): 0.1788 / 0.3080
    [2/5][1050/1583]        Loss_D: 0.5230  Loss_G: 2.9233  D(x): 0.8224    D(G(z)): 0.2455 / 0.0703
    [2/5][1100/1583]        Loss_D: 0.5311  Loss_G: 2.2748  D(x): 0.8222    D(G(z)): 0.2552 / 0.1260
    [2/5][1150/1583]        Loss_D: 0.9006  Loss_G: 3.8327  D(x): 0.9050    D(G(z)): 0.5015 / 0.0295
    [2/5][1200/1583]        Loss_D: 1.0308  Loss_G: 0.5902  D(x): 0.4446    D(G(z)): 0.0680 / 0.5899
    [2/5][1250/1583]        Loss_D: 1.8627  Loss_G: 5.3848  D(x): 0.9468    D(G(z)): 0.7879 / 0.0084
    [2/5][1300/1583]        Loss_D: 0.5449  Loss_G: 1.7229  D(x): 0.7020    D(G(z)): 0.1325 / 0.2218
    [2/5][1350/1583]        Loss_D: 0.5451  Loss_G: 2.0869  D(x): 0.7396    D(G(z)): 0.1875 / 0.1465
    [2/5][1400/1583]        Loss_D: 0.9796  Loss_G: 1.5296  D(x): 0.4621    D(G(z)): 0.0489 / 0.3000
    [2/5][1450/1583]        Loss_D: 1.0319  Loss_G: 1.2058  D(x): 0.4381    D(G(z)): 0.0675 / 0.3517
    [2/5][1500/1583]        Loss_D: 0.8493  Loss_G: 3.0696  D(x): 0.8938    D(G(z)): 0.4619 / 0.0636
    [2/5][1550/1583]        Loss_D: 0.5316  Loss_G: 2.5088  D(x): 0.7957    D(G(z)): 0.2322 / 0.1042
    [3/5][0/1583]   Loss_D: 0.5447  Loss_G: 2.1140  D(x): 0.6783    D(G(z)): 0.0933 / 0.1538
    [3/5][50/1583]  Loss_D: 0.5394  Loss_G: 2.6242  D(x): 0.7692    D(G(z)): 0.2038 / 0.0917
    [3/5][100/1583] Loss_D: 0.7439  Loss_G: 2.4031  D(x): 0.7317    D(G(z)): 0.3011 / 0.1154
    [3/5][150/1583] Loss_D: 0.8023  Loss_G: 3.1821  D(x): 0.8579    D(G(z)): 0.4197 / 0.0572
    [3/5][200/1583] Loss_D: 3.5024  Loss_G: 0.6032  D(x): 0.0468    D(G(z)): 0.0101 / 0.6368
    [3/5][250/1583] Loss_D: 0.6543  Loss_G: 1.9209  D(x): 0.6537    D(G(z)): 0.1567 / 0.1803
    [3/5][300/1583] Loss_D: 0.6316  Loss_G: 1.8408  D(x): 0.6888    D(G(z)): 0.1625 / 0.1967
    [3/5][350/1583] Loss_D: 1.1576  Loss_G: 4.1980  D(x): 0.8957    D(G(z)): 0.5996 / 0.0213
    [3/5][400/1583] Loss_D: 0.5648  Loss_G: 2.9991  D(x): 0.7867    D(G(z)): 0.2343 / 0.0667
    [3/5][450/1583] Loss_D: 0.7732  Loss_G: 1.9517  D(x): 0.6319    D(G(z)): 0.1911 / 0.1785
    [3/5][500/1583] Loss_D: 0.9232  Loss_G: 3.9511  D(x): 0.8900    D(G(z)): 0.4979 / 0.0266
    [3/5][550/1583] Loss_D: 0.6044  Loss_G: 2.8933  D(x): 0.8187    D(G(z)): 0.3016 / 0.0700
    [3/5][600/1583] Loss_D: 0.6475  Loss_G: 1.5161  D(x): 0.6392    D(G(z)): 0.1365 / 0.2637
    [3/5][650/1583] Loss_D: 2.2913  Loss_G: 4.4382  D(x): 0.9328    D(G(z)): 0.8284 / 0.0246
    [3/5][700/1583] Loss_D: 0.5982  Loss_G: 1.8616  D(x): 0.7737    D(G(z)): 0.2528 / 0.1800
    [3/5][750/1583] Loss_D: 0.7478  Loss_G: 2.7308  D(x): 0.7812    D(G(z)): 0.3544 / 0.0810
    [3/5][800/1583] Loss_D: 0.6695  Loss_G: 3.0347  D(x): 0.8363    D(G(z)): 0.3619 / 0.0581
    [3/5][850/1583] Loss_D: 0.5696  Loss_G: 3.2753  D(x): 0.8734    D(G(z)): 0.3192 / 0.0498
    [3/5][900/1583] Loss_D: 0.6227  Loss_G: 1.6989  D(x): 0.6174    D(G(z)): 0.0803 / 0.2262
    [3/5][950/1583] Loss_D: 0.7738  Loss_G: 1.2542  D(x): 0.6267    D(G(z)): 0.2058 / 0.3199
    [3/5][1000/1583]        Loss_D: 1.2990  Loss_G: 4.3936  D(x): 0.9544    D(G(z)): 0.6511 / 0.0195
    [3/5][1050/1583]        Loss_D: 0.7019  Loss_G: 2.2143  D(x): 0.7075    D(G(z)): 0.2463 / 0.1411
    [3/5][1100/1583]        Loss_D: 1.3295  Loss_G: 4.5981  D(x): 0.9558    D(G(z)): 0.6687 / 0.0146
    [3/5][1150/1583]        Loss_D: 0.7469  Loss_G: 1.9873  D(x): 0.7649    D(G(z)): 0.3166 / 0.1696
    [3/5][1200/1583]        Loss_D: 0.6022  Loss_G: 1.5922  D(x): 0.6625    D(G(z)): 0.1367 / 0.2440
    [3/5][1250/1583]        Loss_D: 0.8305  Loss_G: 1.1034  D(x): 0.5673    D(G(z)): 0.1682 / 0.3810
    [3/5][1300/1583]        Loss_D: 1.0482  Loss_G: 3.7645  D(x): 0.9031    D(G(z)): 0.5655 / 0.0328
    [3/5][1350/1583]        Loss_D: 0.5799  Loss_G: 3.3384  D(x): 0.8526    D(G(z)): 0.3130 / 0.0476
    [3/5][1400/1583]        Loss_D: 0.6088  Loss_G: 3.0655  D(x): 0.9199    D(G(z)): 0.3766 / 0.0624
    [3/5][1450/1583]        Loss_D: 0.7549  Loss_G: 2.8069  D(x): 0.7516    D(G(z)): 0.3222 / 0.0839
    [3/5][1500/1583]        Loss_D: 1.2116  Loss_G: 0.6416  D(x): 0.3735    D(G(z)): 0.0589 / 0.5677
    [3/5][1550/1583]        Loss_D: 1.2729  Loss_G: 0.9605  D(x): 0.3445    D(G(z)): 0.0535 / 0.4348
    [4/5][0/1583]   Loss_D: 0.7128  Loss_G: 2.2280  D(x): 0.7539    D(G(z)): 0.3083 / 0.1299
    [4/5][50/1583]  Loss_D: 0.6650  Loss_G: 2.2337  D(x): 0.7127    D(G(z)): 0.2304 / 0.1431
    [4/5][100/1583] Loss_D: 0.6506  Loss_G: 1.5689  D(x): 0.6321    D(G(z)): 0.1168 / 0.2549
    [4/5][150/1583] Loss_D: 0.5558  Loss_G: 2.9658  D(x): 0.8876    D(G(z)): 0.3199 / 0.0709
    [4/5][200/1583] Loss_D: 0.6818  Loss_G: 1.5348  D(x): 0.6476    D(G(z)): 0.1655 / 0.2606
    [4/5][250/1583] Loss_D: 0.5595  Loss_G: 2.1594  D(x): 0.6837    D(G(z)): 0.1260 / 0.1446
    [4/5][300/1583] Loss_D: 0.9177  Loss_G: 1.7132  D(x): 0.6029    D(G(z)): 0.2526 / 0.2272
    [4/5][350/1583] Loss_D: 0.9042  Loss_G: 1.1536  D(x): 0.4830    D(G(z)): 0.0662 / 0.3513
    [4/5][400/1583] Loss_D: 0.5737  Loss_G: 2.5080  D(x): 0.7968    D(G(z)): 0.2557 / 0.1073
    [4/5][450/1583] Loss_D: 0.5612  Loss_G: 1.9423  D(x): 0.7442    D(G(z)): 0.1968 / 0.1706
    [4/5][500/1583] Loss_D: 0.6479  Loss_G: 2.6896  D(x): 0.8124    D(G(z)): 0.3213 / 0.0842
    [4/5][550/1583] Loss_D: 0.5404  Loss_G: 2.1479  D(x): 0.7058    D(G(z)): 0.1366 / 0.1461
    [4/5][600/1583] Loss_D: 0.6495  Loss_G: 2.4230  D(x): 0.8178    D(G(z)): 0.3232 / 0.1149
    [4/5][650/1583] Loss_D: 0.8632  Loss_G: 1.0302  D(x): 0.5168    D(G(z)): 0.0896 / 0.3981
    [4/5][700/1583] Loss_D: 0.6289  Loss_G: 2.2613  D(x): 0.7629    D(G(z)): 0.2666 / 0.1279
    [4/5][750/1583] Loss_D: 2.1401  Loss_G: 1.0056  D(x): 0.2417    D(G(z)): 0.1118 / 0.4185
    [4/5][800/1583] Loss_D: 0.6481  Loss_G: 1.6516  D(x): 0.6887    D(G(z)): 0.1958 / 0.2239
    [4/5][850/1583] Loss_D: 0.7518  Loss_G: 3.4301  D(x): 0.9168    D(G(z)): 0.4439 / 0.0451
    [4/5][900/1583] Loss_D: 0.8320  Loss_G: 1.0878  D(x): 0.5146    D(G(z)): 0.0675 / 0.4041
    [4/5][950/1583] Loss_D: 0.5256  Loss_G: 1.9313  D(x): 0.7134    D(G(z)): 0.1354 / 0.1785
    [4/5][1000/1583]        Loss_D: 1.0996  Loss_G: 4.1374  D(x): 0.9100    D(G(z)): 0.5840 / 0.0231
    [4/5][1050/1583]        Loss_D: 0.8534  Loss_G: 1.0918  D(x): 0.5067    D(G(z)): 0.0656 / 0.3913
    [4/5][1100/1583]        Loss_D: 0.6114  Loss_G: 2.6835  D(x): 0.8625    D(G(z)): 0.3256 / 0.0911
    [4/5][1150/1583]        Loss_D: 1.5442  Loss_G: 0.2662  D(x): 0.2790    D(G(z)): 0.0383 / 0.7897
    [4/5][1200/1583]        Loss_D: 1.0832  Loss_G: 3.7188  D(x): 0.9010    D(G(z)): 0.5687 / 0.0345
    [4/5][1250/1583]        Loss_D: 0.6142  Loss_G: 3.2957  D(x): 0.8538    D(G(z)): 0.3286 / 0.0510
    [4/5][1300/1583]        Loss_D: 0.4714  Loss_G: 2.3520  D(x): 0.7537    D(G(z)): 0.1412 / 0.1291
    [4/5][1350/1583]        Loss_D: 0.5594  Loss_G: 1.6863  D(x): 0.7342    D(G(z)): 0.1754 / 0.2243
    [4/5][1400/1583]        Loss_D: 0.4233  Loss_G: 2.2400  D(x): 0.8577    D(G(z)): 0.2130 / 0.1326
    [4/5][1450/1583]        Loss_D: 0.6392  Loss_G: 3.7684  D(x): 0.8736    D(G(z)): 0.3624 / 0.0302
    [4/5][1500/1583]        Loss_D: 0.6295  Loss_G: 1.9195  D(x): 0.6754    D(G(z)): 0.1719 / 0.1772
    [4/5][1550/1583]        Loss_D: 0.6141  Loss_G: 2.0106  D(x): 0.6374    D(G(z)): 0.1047 / 0.1655


Results
-------

Finally, lets check out how we did. Here, we will look at three
different results. First, we will see how D and G’s losses changed
during training. Second, we will visualize G’s output on the fixed_noise
batch for every epoch. And third, we will look at a batch of real data
next to a batch of fake data from G.

**Loss versus training iteration**

Below is a plot of D & G’s losses versus training iterations.



.. code-block:: default


    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()





.. image:: /beginner/images/sphx_glr_dcgan_faces_tutorial_002.png
    :class: sphx-glr-single-img




**Visualization of G’s progression**

Remember how we saved the generator’s output on the fixed_noise batch
after every epoch of training. Now, we can visualize the training
progression of G with an animation. Press the play button to start the
animation.



.. code-block:: default


    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())





.. image:: /beginner/images/sphx_glr_dcgan_faces_tutorial_003.png
    :class: sphx-glr-single-img




**Real Images vs. Fake Images**

Finally, lets take a look at some real images and fake images side by
side.



.. code-block:: default


    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()





.. image:: /beginner/images/sphx_glr_dcgan_faces_tutorial_004.png
    :class: sphx-glr-single-img




Where to Go Next
----------------

We have reached the end of our journey, but there are several places you
could go from here. You could:

-  Train for longer to see how good the results get
-  Modify this model to take a different dataset and possibly change the
   size of the images and the model architecture
-  Check out some other cool GAN projects
   `here <https://github.com/nashory/gans-awesome-applications>`__
-  Create GANs that generate
   `music <https://deepmind.com/blog/wavenet-generative-model-raw-audio/>`__



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 15 minutes  0.260 seconds)


.. _sphx_glr_download_beginner_dcgan_faces_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: dcgan_faces_tutorial.py <dcgan_faces_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: dcgan_faces_tutorial.ipynb <dcgan_faces_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
