# Modified by Alex Hong, 2015
# nn implementation
# thealexhong@gmail.com

# Code by Navdeep Jaitly, 2013.
# Email: ndjaitly@gmail.com

from nnet_layers import *
import sys, logging, os, scipy.io
from numpy import zeros, savez, log
 
# create logger
logger = logging.getLogger('nnet_train')
logger.setLevel(logging.INFO)
# create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


class nn(object):
    def __init__(self):
        pass

    def save(self, file_name):
        params_dict = {} 
        params_dict['lst_layer_names'] = [layer.name for layer in self._lst_layers]
        params_dict['lst_layer_type'] = self._lst_layer_type
        params_dict['lst_num_hid'] = self._lst_num_hid
        params_dict['data_dim'] = self._data_dim

        for layer in self._lst_layers:
            layer.add_params_to_dict(params_dict)

        scipy.io.savemat(file_name, params_dict)
        #util.save(file_name, " ".join(params_dict.keys()), params_dict)

    def load(self, file_name):
        params_dict = scipy.io.loadmat(file_name)
        self._lst_layer_type, self._lst_num_hid, self._data_dim = \
                                           params_dict['lst_layer_type'], \
                                               params_dict['lst_num_hid'], \
                                                   params_dict['data_dim']

        if not hasattr(self, '_lst_layers'):
            logging.info("Creating new layers from parameters in file: %s"%file_name)
            self._lst_layers = [] 
            for (layer_name, layer_type) in zip(params_dict['lst_layer_names'],
                                                self._lst_layer_type[0]):
                layer = create_empty_nnet_layer(layer_name, layer_type)
                layer.copy_params_from_dict(params_dict)
                self._lst_layers.append(layer)
        else:
            logging.info("Updating layer parameters using file: %s"%file_name)
            for layer in self._lst_layers:
                layer.copy_params_from_dict(params_dict)

        self.num_layers = len(self._lst_layers)


    def get_num_layers(self):
        return len(self._lst_layers)

    def get_code_dim(self):
        return self._lst_num_hid[-1]

    def create_nnet_from_def(self, lst_def):
        self._layers = [] 
        self.num_layers = len(lst_def)

        self._data_dim = lst_def[0].input_dim

        self._lst_num_hid = []
        self._lst_layer_type = []
        self._lst_layers = []

        for layer_num, layer_def in enumerate(lst_def):
            self._lst_num_hid.append(layer_def.num_units)
            self._lst_layer_type.append(layer_def.layer_type)
            self._lst_layers.append(create_nnet_layer(layer_def))


    def fwd_prop(self, data):
        """
        Forward propagation logic across the NN
        """

        lst_layer_outputs = []
        current_layer_output = data

        for layer in self._lst_layers:
            current_layer_output = layer.fwd_prop(current_layer_output)
            lst_layer_outputs.append(current_layer_output)

        return lst_layer_outputs


    def back_prop(self, lst_layer_outputs, data, targets):
        """
        Back propagation logic across the NN
        """

        layers_outputs = lst_layer_outputs[::-1]
        layers = self._lst_layers[::-1]

        prev_layers_outputs = lst_layer_outputs[::-1][1:]
        prev_layers_outputs.append(data)

        output_grad = 0

        for layer, layer_outputs, prev_layer_outputs in zip(layers, layers_outputs, prev_layers_outputs):
            if layer.is_softmax():
                act_grad = layer.compute_act_gradients_from_targets(targets, layer_outputs)
                input_grad = layer.back_prop(act_grad, prev_layer_outputs)
                output_grad = input_grad
            else:
                act_grad = layer.compute_act_grad_from_output_grad(layer_outputs, output_grad)
                input_grad = layer.back_prop(act_grad, prev_layer_outputs)
                output_grad = input_grad

    def apply_gradients(self, eps, momentum, l2, batch_size):
        """
        Gradient update logic across the Neural Network.
        """

        for layer in self._lst_layers:
            layer.apply_gradients(momentum, eps, l2, batch_size)


    def create_predictions(self, data_src):
        """
        Function used to create predictions from acoustics.
        """
        num_pts, classif_err_sum, lg_p_sum = 0, 0, 0

        pred_lst = []
        num_output_frames = 0
        for utt_num  in range(data_src.get_num_utterances()):
            data = data_src.get_utterance_data(utt_num, 
                                               get_labels=False)
            lst_layer_outputs = self.fwd_prop(data)
            pred_lst.append(log(1e-32 + lst_layer_outputs[-1]))
            num_output_frames += pred_lst[-1].shape[1]

        
        return pred_lst, num_output_frames


    def test(self, data_src):
        """
        Function used to test accuracy.
        """
        num_pts, classif_err_sum, lg_p_sum = 0, 0, 0

        for utt_num  in range(data_src.get_num_utterances()):
            data, label_mat = data_src.get_utterance_data(utt_num)
            num_pts += data.shape[1]
            lst_layer_outputs = self.fwd_prop(data)
            num_correct, log_prob = self._lst_layers[-1].compute_accuraccy(\
                                            lst_layer_outputs[-1], label_mat)
            classif_err_sum += (data.shape[1] - num_correct)
            lg_p_sum += log_prob

        classif_err = classif_err_sum*100./num_pts
        logging.info("TESTING Classif Err = %.3f, lg(p) %.4f\n"%(\
                         classif_err, lg_p_sum*1./num_pts))
        sys.stderr.write("TESTING Classif Err = %.3f, lg(p) %.4f\n"%(\
                         classif_err, lg_p_sum*1./num_pts))
        #logging.info("%.3f\n"%(\
        #                 classif_err))
        #sys.stderr.write("%.3f\n"%(\
        #                 classif_err))
        sys.stderr.flush()
        ch.flush()

        return 100 - classif_err, lg_p_sum*1./num_pts

    def train_for_one_epoch(self, data_src, eps, momentum, l2, batch_size):
        '''
        Work horse of the learning for one epoch. As long as the other
        functions are working correctly, and satisfy the interface, 
        there should be no need to change this function. 
        '''
        try:
            self.__cur_epoch += 1
        except AttributeError:
            self.__cur_epoch = 1

        try:
            self._tot_batch
        except AttributeError:
            self._tot_batch = 0

        num_pts, classif_err_sum, lg_p_sum = 0, 0, 0
        batch = 0

        for  (data, label_mat) in data_src.get_iterator(batch_size):
            batch += 1
            num_pts += batch_size
            lst_layer_outputs = self.fwd_prop(data)
            num_correct, log_prob = self._lst_layers[-1].compute_accuraccy(\
                                            lst_layer_outputs[-1], label_mat)
            classif_err_sum += (data.shape[1] - num_correct)
            lg_p_sum += log_prob

            self.back_prop(lst_layer_outputs, data, label_mat)
            self.apply_gradients(eps, momentum, l2, batch_size)
            self._tot_batch += 1
        
        classif_err = classif_err_sum*100./num_pts
        logging.info("Epoch = %d, batch = %d, Classif Err = %.3f, lg(p) %.4f"%(\
                   self.__cur_epoch, batch, classif_err, lg_p_sum*1./num_pts))
        sys.stderr.write("Epoch = %d, batch = %d, Classif Err = %.3f, lg(p) %.4f\n"%(\
                   self.__cur_epoch, batch, classif_err, lg_p_sum*1./num_pts))
        #logging.info("%.3f\n"%(\
        #           classif_err))
        #sys.stderr.write("%.3f\n"%(\
        #           classif_err))
        sys.stderr.flush()
        ch.flush()

    def should_stop_training(self, epoch, stopper, valid_src):
        """
        Check if the training process should stop (prevent NN from overfitting)
        """

        validation_accuracy, log_p = self.test(valid_src)
        validation_error = 100 - validation_accuracy

        # early stopping strategy for controlling over-fitting
        return stopper.should_early_stop(epoch, validation_error)


class early_stopper:
    """
    Early stopping implementation
    """

    def __init__(self):
        self.epochs_to_consider = 3
        self.occurrence = 0
        self.best_epoch = 0
        self.best_validation_error = 100

    def should_early_stop(self, epoch, validation_error):
        """
        Check the need of stopping training according to validation error
        """

        should_stop = False
        if validation_error >= self.best_validation_error:
            self.occurrence += 1
            if self.occurrence == self.epochs_to_consider:
                print
                print "Iteration stopped at epoch %d" % epoch
                print "Best validation error found at epoch %d" % self.best_epoch
                print "Best validation error is %f" % self.best_validation_error
                should_stop = True
        else:
            self.occurrence = 0
            self.best_epoch = epoch
            self.best_validation_error = validation_error
        return should_stop
