import tensorflow as tf
import numpy as np
import yaml
from abc import abstractmethod


activation_dict = {"relu": tf.nn.relu, None: None, "None": None}

# module confEstimator
class TensorDict(dict):
    """
    Class that contains a dictionary of tensors
    """
    def last(self):
        return self.get(list(self.keys())[-1])
    
    def get_many(self, t_list):
        return {x: self.get(x) for x in t_list}
    
    def summary(self):
        print("** Specified Tensor Architecture **")
        for k, v in self.items():
            print("{}: {}".format(k, v))       
        print("**")
        

def create_conv_layer(tensor_in, spec, name, mode=tf.estimator.ModeKeys.TRAIN):
    return {name: tf.layers.conv2d(inputs=tensor_in, 
                                   filters=spec['filters'],
                                   kernel_size=spec['kernel_size'],
                                   padding=spec['padding'],
                                   activation=activation_dict[spec['activation']],
                                   name=name)}

def create_maxpool_layer(tensor_in, spec, name, mode=tf.estimator.ModeKeys.TRAIN):
    return {name: tf.layers.max_pooling2d(inputs=tensor_in,
                                          pool_size=spec['pool_size'],
                                          strides=spec['strides'],
                                          name=name)}

def create_dense_layer(tensor_in, spec, name, mode=tf.estimator.ModeKeys.TRAIN):
    # creates an extra layer that must be flattened as well
    out = {}
    out[name+'_flatten'] = tf.reshape(tensor_in, [-1, np.prod(tensor_in.shape[1:4])])
    out[name] = tf.layers.dense(inputs=out[name+'_flatten'],
                                units=spec['units'],
                                activation=activation_dict[spec['activation']],
                                name=name)
    return out
                      
def create_dropout_layer(tensor_in, spec, name, mode=tf.estimator.ModeKeys.TRAIN):
    return {name: tf.layers.dropout(inputs=tensor_in, rate=spec['rate'], training = mode == tf.estimator.ModeKeys.TRAIN) }

l_dict = {'CONV':create_conv_layer, 'POOL':create_maxpool_layer, 'FC': create_dense_layer, 'DROP': create_dropout_layer}

def tensorboard_summaries(conv_layers=None, dense_layers=None):
    if conv_layers is not None:
        for cl_name, cl in conv_layers.items():
            shape = cl.shape.as_list()
            for f in range(shape[3]):
                tf.summary.image("{}/filter/{}".format(cl_name, f), tf.reshape(cl[:, :, :, f], [tf.shape(cl)[0], shape[1], shape[2], 1]))  
    if dense_layers is not None:
        for dl_name, dl in dense_layers.items():
            tf.summary.histogram("{}/weights".format(dl_name), dl) 


class confEstimator(tf.estimator.Estimator):
    """
    Class that allows building a certain neural-net based on a configuration YAML file. It must be overcharged.
    """
            
    def __init__(self, conf_file, **kwargs):
        self.conf_file = conf_file
        with open(conf_file) as f:
            self.conf = yaml.load(f)
        super(confEstimator, self).__init__(self.conf_model_fn, **kwargs)
        
    @abstractmethod
    def conf_model_fn(self, features, labels, model):
        pass
    
    def _get_tensor_dict(self, data_in, mode):
        tensor_dict = TensorDict()  # will contain all the information on the layers and tensors
        # if data_in is a dictionary (e.g. 'x'), unroll
        if type(data_in)==dict:  # conver to np.array
            shaped_data_in = np.array([data_in[k] for k in data_in.keys()])
            if len(data_in.keys())==1:  # remove unncessary dimension
                shaped_data_in = shaped_data_in[0]
        else:
            shaped_data_in = data_in
        num_channels = 1 if len(shaped_data_in.shape)==3 else shaped_data_in.shape[3]
        tensor_dict['input_layer'] = tf.reshape(shaped_data_in, [-1, shaped_data_in.shape[1], shaped_data_in.shape[2], num_channels]) 
        for l_name, l_def in self.conf['layers'].items():
            tensor_dict.update(l_dict[l_def['type']](tensor_dict.last(), l_def['specs'], l_name, mode))
        return tensor_dict
    
    def get_layers_of_type(self, l_type):
        return [l_name for l_name, l_def in self.conf['layers'].items() if l_def['type']==l_type]
    

class confRegressorEstimator(confEstimator):
    """
    Class that allows for a image classification based on a neural-net which architecture is on a YAML file
    """
    def conf_model_fn(self, features, labels, mode):

        # tensor_dict
        td =  self._get_tensor_dict(features, mode)
        #td.summary()
        
        # tensorboard summary for weights and variables in the layers   
        tensorboard_summaries(conv_layers=td.get_many(self.get_layers_of_type('CONV')), 
                             dense_layers=td.get_many(self.get_layers_of_type('FC')))

        # predictions
        predictions = {
            "predictions": td.last()  # return the last predicted image
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            # wrap predictions into a class and return EstimatorSpec object
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
        # minimization function
        loss = tf.losses.mean_squared_error(labels=labels, predictions=td.last())  # loss is a scalar tensor
        
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.conf['learning_rate'])
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["predictions"]),
            "mean_absolute_error": tf.metrics.mean_absolute_error(labels=labels, predictions=predictions["predictions"]), 
            "RMS": tf.metrics.root_mean_squared_error(labels=labels, predictions=predictions["predictions"])
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


class confClassifierEstimator(confEstimator):
    """
    Class that allows for a image classification based on a convolutional neural-net which architecture is on a YAML file
    """
    def conf_model_fn(self, features, labels, mode):

        # tensor_dict
        td =  self._get_tensor_dict(features, mode)
        
        # tensorboard summary for weights and variables in the layers   
        tensorboard_summaries(conv_layers=td.get_many(self.get_layers_of_type('CONV')), 
                             dense_layers=td.get_many(self.get_layers_of_type('FC')))

        # predictions
        predictions = {
            "classes": tf.argmax(input=td.last(), axis=1), 
            "probabilities": tf.nn.softmax(td.last(), name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            # wrap predictions into a class and return EstimatorSpec object
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # one-hot encoding
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=td.last().shape[1])

        # minimize on cross-entropy
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=td.last())  # loss is a scalar tensor

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.conf['learning_rate'])
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
            "precision": tf.metrics.precision(labels=labels, predictions=predictions["classes"]),
            "recall": tf.metrics.recall(labels=labels, predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
