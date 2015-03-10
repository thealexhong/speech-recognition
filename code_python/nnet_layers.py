# Modified by Alex Hong, 2015
# nn implementation
# thealexhong@gmail.com

# Code by Navdeep Jaitly, 2013
# Email: ndjaitly@gmail.com

from numpy import sqrt, isnan, Inf, dot, zeros, exp, log, sum, newaxis, vstack
from numpy.random import randn


SIGMOID_LAYER = 0
SOFTMAX_LAYER = 1

class layer_definition(object):
    def __init__(self, name, layer_type, input_dim, num_units, wt_sigma):
        self.name, self.layer_type, self.input_dim, self.num_units, \
          self.wt_sigma  =  name, layer_type, input_dim, num_units, wt_sigma


class layer(object):
    def __init__(self, name):
        self.name = name

    @property
    def shape(self):
        return self._wts.shape

    @property
    def num_hid(self):
        return self._wts.shape[1]

    @property
    def num_dims(self):
        return self._wts.shape[0]

    def create_params(self, layer_def):
        input_dim, output_dim, wt_sigma = layer_def.input_dim, \
                        layer_def.num_units, layer_def.wt_sigma

        self._wts = randn(input_dim, output_dim) * wt_sigma
        self._b = zeros((output_dim, 1))

        self._wts_grad = zeros(self._wts.shape)
        self._wts_inc = zeros(self._wts.shape)

        self._b_grad = zeros(self._b.shape)
        self._b_inc = zeros(self._b.shape)

        self.__num_params = input_dim * output_dim

    def add_params_to_dict(self, params_dict):
        params_dict[self.name + "_wts"] = self._wts.copy()
        params_dict[self.name + "_b"] = self._b.copy()

    def copy_params_from_dict(self, params_dict):
        self._wts = params_dict[self.name + "_wts"].copy()
        self._b = params_dict[self.name + "_b"].copy()
        self.__num_params = self._wts.shape[0] * self._wts.shape[1]
        self._wts_inc = zeros(self._wts.shape)
        self._b_inc = zeros(self._b.shape)

    def apply_gradients(self, momentum, eps, l2, batch_size):
        """ 
        Gradient update for the layer
        """
        w_momentum = momentum * self._wts_inc
        b_momentum = momentum * self._b_inc
        w_learning = - (self._wts_grad * eps / batch_size)
        b_learning = - (self._b_grad * eps / batch_size)
        w_l2 = - (l2 * self._wts * eps / batch_size)
        b_l2 = - (l2 * self._b * eps / batch_size)
        self._wts_inc = w_learning + w_l2 + w_momentum
        self._b_inc = b_learning + b_l2 + b_momentum
        self._wts += self._wts_inc
        self._b += self._b_inc

    def back_prop(self, act_grad, prev_layer_outputs):
        """
        Back propagation for the layer
        """
        self._wts_grad = dot(prev_layer_outputs, act_grad.T)
        self._b_grad = act_grad.sum(1)[:, newaxis]
        input_grad = dot(self._wts, act_grad)

        return input_grad
 

class sigmoid_layer(layer):
    """
    Sigmoid implementation of a NN layer (hidden units layer)
    """

    def fwd_prop(self, data):
        a = dot(data.T, self._wts) + self._b.T
        outputs = self.sigmoid(a)

        return outputs.T

    def compute_act_grad_from_output_grad(self, layer_outputs, output_grad):
        act_grad = self.dsigmoid(layer_outputs)

        return act_grad * output_grad

    def sigmoid(self, x):
        return 1.0 / (1 + exp(-x))

    def dsigmoid(self, x):
        return x * (1.0 - x)

    def is_softmax(self):
        return False
 
class softmax_layer(layer):
    """
    Sofmax implementation of a NN layer (output layer).
    """

    def fwd_prop(self, data):
        a = dot(data.T, self._wts) + self._b.T
        outputs = self.softmax(a)

        return outputs.T

    def compute_act_gradients_from_targets(self, targets, output):
        act_grad = output - targets

        return act_grad

    def softmax(self, x):
        exp_x_sum = vstack(exp(x).sum(1))

        return 1.0 * exp(x) / exp_x_sum

    def is_softmax(self):
        return True

    @staticmethod 
    def compute_accuraccy(probs, label_mat):
        num_correct = sum(probs.argmax(axis=0) == label_mat.argmax(axis=0))
        log_probs = sum(log(probs) * label_mat)
        return num_correct, log_probs

def create_empty_nnet_layer(name, layer_type):
    if layer_type == SIGMOID_LAYER:
        layer = sigmoid_layer(name)
    elif layer_type == SOFTMAX_LAYER:
        layer = softmax_layer(name)
    else:
        raise Exception, "Unknown layer type"
    return layer

def create_nnet_layer(layer_def):
    layer = create_empty_nnet_layer(layer_def.name, layer_def.layer_type)
    layer.create_params(layer_def)
    return layer
