.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_advanced_numpy_extensions_tutorial.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_advanced_numpy_extensions_tutorial.py:


Creating Extensions Using numpy and scipy
=========================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_

**Updated by**: `Adam Dziedzic <https://github.com/adam-dziedzic>`_

In this tutorial, we shall go through two tasks:

1. Create a neural network layer with no parameters.

    -  This calls into **numpy** as part of its implementation

2. Create a neural network layer that has learnable weights

    -  This calls into **SciPy** as part of its implementation

.. code-block:: default


    import torch
    from torch.autograd import Function







Parameter-less example
----------------------

This layer doesn’t particularly do anything useful or mathematically
correct.

It is aptly named BadFFTFunction

**Layer Implementation**


.. code-block:: default


    from numpy.fft import rfft2, irfft2


    class BadFFTFunction(Function):
        @staticmethod
        def forward(ctx, input):
            numpy_input = input.detach().numpy()
            result = abs(rfft2(numpy_input))
            return input.new(result)

        @staticmethod
        def backward(ctx, grad_output):
            numpy_go = grad_output.numpy()
            result = irfft2(numpy_go)
            return grad_output.new(result)

    # since this layer does not have any parameters, we can
    # simply declare this as a function, rather than as an nn.Module class


    def incorrect_fft(input):
        return BadFFTFunction.apply(input)







**Example usage of the created layer:**


.. code-block:: default


    input = torch.randn(8, 8, requires_grad=True)
    result = incorrect_fft(input)
    print(result)
    result.backward(torch.randn(result.size()))
    print(input)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    tensor([[ 4.8203,  5.8899, 14.4428,  2.1572,  5.2869],
            [10.2004,  7.4153,  7.3944,  9.3021,  6.5521],
            [12.1627, 10.8466,  5.7293,  3.7987,  4.6167],
            [ 8.8490,  4.8005,  4.3963, 16.5978,  3.9824],
            [ 4.6766,  7.2633,  8.8207,  2.2980,  8.6458],
            [ 8.8490,  4.0130,  0.8623,  9.4462,  3.9824],
            [12.1627,  4.7612,  3.2154,  5.2606,  4.6167],
            [10.2004,  8.7563,  8.1784, 10.5274,  6.5521]],
           grad_fn=<BadFFTFunctionBackward>)
    tensor([[ 0.0767,  2.1941, -0.5802,  1.6767, -0.3548, -1.2863,  1.6679,  0.3904],
            [ 0.0719, -0.8389,  2.0275, -1.0302, -0.3114,  1.3759,  1.6271,  0.1921],
            [-1.0425, -2.1732, -0.1290, -0.1667,  1.1759,  0.0763, -0.3798, -0.3751],
            [-0.3812, -1.8373, -0.1515, -0.6597, -0.7138, -1.3929,  0.0812,  0.6378],
            [-0.5288, -2.3403, -0.4517, -0.8640, -0.2272,  0.3074, -1.0100,  1.6383],
            [-0.4848,  0.0397, -0.5262,  0.3533,  0.0473, -0.7294,  0.9737,  0.9096],
            [ 0.6830, -0.7077,  0.3878,  2.3164,  0.1531,  0.1016, -0.3160,  0.0159],
            [-0.0621, -0.6109,  0.1377, -0.5859, -1.1776, -1.1818, -0.0491, -0.4988]],
           requires_grad=True)


Parametrized example
--------------------

In deep learning literature, this layer is confusingly referred
to as convolution while the actual operation is cross-correlation
(the only difference is that filter is flipped for convolution,
which is not the case for cross-correlation).

Implementation of a layer with learnable weights, where cross-correlation
has a filter (kernel) that represents weights.

The backward pass computes the gradient wrt the input and the gradient wrt the filter.


.. code-block:: default


    from numpy import flip
    import numpy as np
    from scipy.signal import convolve2d, correlate2d
    from torch.nn.modules.module import Module
    from torch.nn.parameter import Parameter


    class ScipyConv2dFunction(Function):
        @staticmethod
        def forward(ctx, input, filter, bias):
            # detach so we can cast to NumPy
            input, filter, bias = input.detach(), filter.detach(), bias.detach()
            result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
            result += bias.numpy()
            ctx.save_for_backward(input, filter, bias)
            return torch.as_tensor(result, dtype=input.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            grad_output = grad_output.detach()
            input, filter, bias = ctx.saved_tensors
            grad_output = grad_output.numpy()
            grad_bias = np.sum(grad_output, keepdims=True)
            grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
            # the previous line can be expressed equivalently as:
            # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
            grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
            return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)


    class ScipyConv2d(Module):
        def __init__(self, filter_width, filter_height):
            super(ScipyConv2d, self).__init__()
            self.filter = Parameter(torch.randn(filter_width, filter_height))
            self.bias = Parameter(torch.randn(1, 1))

        def forward(self, input):
            return ScipyConv2dFunction.apply(input, self.filter, self.bias)








**Example usage:**


.. code-block:: default


    module = ScipyConv2d(3, 3)
    print("Filter and bias: ", list(module.parameters()))
    input = torch.randn(10, 10, requires_grad=True)
    output = module(input)
    print("Output from the convolution: ", output)
    output.backward(torch.randn(8, 8))
    print("Gradient for the input map: ", input.grad)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Filter and bias:  [Parameter containing:
    tensor([[ 0.2263,  0.7514,  0.6118],
            [ 0.2762,  0.3257,  0.9966],
            [-1.1289, -0.7352,  1.6498]], requires_grad=True), Parameter containing:
    tensor([[-0.1789]], requires_grad=True)]
    Output from the convolution:  tensor([[-3.8642,  2.4182,  4.8134, -1.1201, -3.8960,  1.9243,  1.7341,  0.0405],
            [-3.7977,  0.4532,  3.6011, -2.5881, -6.0355,  0.1332,  1.4165, -1.1061],
            [-1.6578, -1.7195,  1.0423,  4.0383, -4.3558, -2.4489, -1.3551, -0.1161],
            [-2.9814, -0.0865,  2.8959, -0.3437, -1.6020, -0.7034, -4.4024, -1.1790],
            [-1.1116, -2.8362,  2.3232, -2.4496, -0.3943,  0.2908, -1.5826, -5.3084],
            [ 2.1403, -3.9328, -1.6491, -0.4587,  0.0607, -0.0718,  0.9385, -6.0518],
            [ 1.5057,  1.4542, -1.8459,  2.1244, -1.8404, -1.5713,  0.4140, -1.0619],
            [-2.6813,  1.5879, -0.3191,  1.6349, -0.4821,  0.5570,  1.3837, -0.3176]],
           grad_fn=<ScipyConv2dFunctionBackward>)
    Gradient for the input map:  tensor([[-4.5311e-02, -1.0415e-02,  6.1443e-01,  1.2965e+00,  8.1650e-01,
              2.4253e-01,  5.3026e-01,  8.6179e-01,  9.6663e-01,  5.1492e-01],
            [ 1.3802e-01,  9.8987e-01,  1.5716e+00,  1.4869e+00,  1.1401e+00,
              3.2342e-02,  9.0484e-01,  2.2459e+00,  1.2841e+00,  6.9099e-01],
            [ 3.4588e-01, -7.2418e-01, -2.4883e+00,  2.8994e-01,  1.1238e+00,
             -4.1463e-01, -1.6830e+00, -2.6074e-01,  9.7712e-01,  1.2714e+00],
            [-1.0701e+00, -2.3785e+00, -2.2895e-01, -9.0837e-01,  3.3138e-01,
              3.0700e-01, -2.9232e+00,  2.1497e+00,  2.9324e+00,  5.7000e-01],
            [ 8.3783e-01,  2.8975e+00,  3.9482e-01, -2.1455e+00,  6.2464e-01,
              1.4041e+00,  1.1941e+00,  1.8894e-01,  3.3435e-01,  1.6298e+00],
            [ 3.4470e-01,  1.4115e+00,  3.4748e+00, -1.5531e+00, -3.8355e+00,
              6.8362e-01, -2.8620e+00, -1.5768e+00,  1.3737e+00,  2.8438e+00],
            [-8.3043e-01, -6.1290e-01,  1.3158e+00, -2.4948e+00, -2.1072e+00,
              1.8696e+00, -3.3428e-01, -1.3286e+00, -3.0442e-01,  9.3658e-01],
            [-1.5169e+00, -2.0771e+00,  1.4435e+00,  1.9215e+00, -2.4033e+00,
             -1.6409e+00,  7.2213e-01, -1.3341e-01, -1.2832e+00,  1.4969e+00],
            [ 2.3948e-01, -1.2789e-01,  1.1106e-03, -1.2216e+00, -1.6312e+00,
              2.7085e+00, -7.2421e-01, -9.8894e-01,  2.1685e+00, -6.4568e-01],
            [ 5.9246e-01,  2.0371e+00, -1.5871e-01, -3.2983e+00,  2.4293e-01,
              5.2833e-01, -6.6200e-01,  3.5649e-01,  5.5168e-02,  3.8935e-01]])


**Check the gradients:**


.. code-block:: default


    from torch.autograd.gradcheck import gradcheck

    moduleConv = ScipyConv2d(3, 3)

    input = [torch.randn(20, 20, dtype=torch.double, requires_grad=True)]
    test = gradcheck(moduleConv, input, eps=1e-6, atol=1e-4)
    print("Are the gradients correct: ", test)




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Are the gradients correct:  True



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.454 seconds)


.. _sphx_glr_download_advanced_numpy_extensions_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: numpy_extensions_tutorial.py <numpy_extensions_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: numpy_extensions_tutorial.ipynb <numpy_extensions_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
