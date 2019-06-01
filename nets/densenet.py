"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current

def transition_op(current, num_outputs, scope='transition'):
    # Here we can use the compression rate 0.5 to decrease the feature map number
    current = slim.conv2d(current, num_outputs, [1,1],padding='SAME', scope=scope + '_conv')
    current = slim.avg_pool2d(current, [2,2], padding='VALID',scope=scope + 'AvgPool')
    return current

def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            ##########################
            # Put your code here.
            ##########################
            # 下面加入自己写的代码。
            # Define the first stage, conv+pooling
            # 16*224*224*3
            # W is width or height.
            # N=(W-F+2P)/S +1
            end_point = 'Conv2d_1a_7x7'
            # 64 kerenel here later will check if can be changed
            net = slim.conv2d(images, 64, [7, 7], stride=2, padding='SAME', scope=end_point)
            end_points[end_point] = net

            print("1-Conv2d_1a_7x7", net.shape)
            # output is (224-7+6)/2 + 1= 112
            # 16*112*112*64
            # Pooling 后大小(W-F +2*P)/S+1

            end_point = 'MaxPool_2a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope=end_point)
            end_points[end_point] = net

            print("1-Pool", net.shape)
            # (112-3 + 1*2)/2 + 1= 55+1= 56
            # Output is
            # 16*56*56*64

            # Define the 2nd stage, Block definition.
            # For block, it will include bottle neck and BN+RELU+Conv+dropout
            # Define the number output of the features as growth.
            # Because growth is the increment by layer inside the block
            # with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
            # stride=1, padding='SAME'):
            # First block layer
            end_point = 'Block_6'
            net = block(net, 6, growth, scope=end_point)
            end_points[end_point] = net

            # Block will not change the size
            # Output is 16*56*56*[64+24*(6-1)]
            print("block1", net.shape)

            # Output is 16*56*56*[64+24*(6-1)] = 16*56*56*184
            # Get the last dim of the tensor and do compression
            num_outputs = reduce_dim(net)

            # Transition layer1 [1*1] plus [2*2] pooling stride = 2
            # bottle neck will decrease the feature map number half
            end_point = 'transition1'
            net = transition_op(net, num_outputs, scope=end_point)
            end_points[end_point] = net
            # [56-2+2*0]/2 + 1= 28 VALID pooling
            # Output is 16*28*28*92
            print("transition1", net.shape)
            # Do block/transition and block as the document said.
            # Block 24 layer
            end_point = 'block2'
            net = block(net, 24, growth, scope=end_point)
            end_points[end_point] = net
            # Output is 16*28*28*[92+24*(24-1)] = 16*28*28*644
            print("block2", net.shape)
            # transistion layer reduce the output feature map and pooling get the size decreased half
            num_outputs = reduce_dim(net)
            end_point = 'transition2'
            net = transition_op(net, num_outputs, scope=end_point)
            end_points[end_point] = net
            # Output is 16*14*14*322
            print("transition2", net.shape)
            # Block
            end_point = 'block3'
            net = block(net, 16, growth, scope=end_point)
            end_points[end_point] = net
            # Output is 16*7*7*[322+24*(16-1)]=16*7*7*682
            print("block3", net.shape)

    with tf.variable_scope('Logits'):
        # Global average pooling.
        # Output [16*7*7*682]
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='GlobalPool')
        end_points['global_pool'] = net

        # Output [16*1*1*682]
        end_point = 'Conv2d_1c_1x1'
        net = slim.conv2d(net, 1000, [1, 1], activation_fn=None,
                          normalizer_fn=None, padding='SAME', scope=end_point)
        end_points[end_point] = net
        # Output [16*1*1*1000]
        end_point = 'Conv2d_1d_1x1'
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, padding='SAME', scope=end_point)
        end_points[end_point] = logits
        print("Conv2d_1d_1x1", logits.shape)
        # Output [16*1*1*10]
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # Output [16*10]
        print("logits1", logits.shape)

    end_points['Logits'] = logits

    # end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    end_points['Predictions'] = slim.softmax(logits, scope='Predictions')

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
