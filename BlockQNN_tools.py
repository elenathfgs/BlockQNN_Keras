import numpy
from keras.layers import Input, Activation, Conv2D, BatchNormalization, \
    MaxPooling2D, AveragePooling2D, Concatenate, Add, GlobalAveragePooling2D, Dense
from keras import initializers, regularizers
import numpy as np
from tensorflow import Tensor
import os


class QNN_layers:
    def __init__(self):
        None

    def input_layer(self, input_shape, name_prefix):
        return Input(shape=input_shape, name="{0}_input".format(name_prefix))

    # pre-activation convolution (ReLU-conv-BN)
    def PCC_conv(self, inputs, name_prefix, kernel_size,
                 current_filters, strides=(1, 1)):
        x = Activation("relu", name="{0}_relu".format(name_prefix))(inputs)

        x = Conv2D(
            filters=current_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer=initializers.he_normal(),
            name='{0}_conv2d'.format(name_prefix)
        )(x)

        x = BatchNormalization(name='{0}_batchNorm'.format(name_prefix))(x)
        return x

    # no down sample inner block
    def Max_pooling(self, inputs, name_prefix, kernel_size,
                    strides=(1, 1)):
        x = MaxPooling2D(
            pool_size=kernel_size,
            padding='same',
            strides=strides,
            name='{0}_maxPool{1}x{1}'.format(name_prefix, kernel_size[0])
        )(inputs)

        return x

    def Average_pooling(self, inputs, name_prefix, kernel_size,
                        strides=(1, 1)):
        x = AveragePooling2D(
            pool_size=kernel_size,
            padding='same',
            strides=strides,
            name='{0}_averagePool{1}x{1}'.format(name_prefix, kernel_size[0])
        )(inputs)

        return x

    def identity(self, inputs, name_prefix):
        x = Activation(
            "linear", name='{0}_linear'.format(name_prefix))(inputs)
        return x

    # when use the add operation, the channel size should be equal
    def adjust_depth(self, inputs, name_prefix, current_filters):
        x = Activation('relu', name='{0}_relu'.format(name_prefix))(inputs)
        x = Conv2D(
            filters=current_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            kernel_initializer=initializers.he_normal(),
            kernel_regularizer=regularizers.l2(),
            name='{0}_conv2d'.format(name_prefix)
        )(x)
        x = BatchNormalization(name="{0}_batchNorm".format(name_prefix))(x)

        return x

    def concat(self, name_prefix, input_list, current_filters):
        x = Concatenate(name='{0}_concat'.format(name_prefix))(input_list)

        # adjust depth after each concat operation to ensure the layer-depth doesn't change
        x = self.adjust_depth(x, name_prefix,
                              current_filters=current_filters)
        return x

    def elemental_add(self, name_prefix, x1, x2):
        assert isinstance(x1, Tensor)
        assert isinstance(x2, Tensor)
        shape_x1 = x1.get_shape().as_list()
        shape_x2 = x2.get_shape().as_list()
        if shape_x1[-1] > shape_x2[-1]:
            currentFilters = shape_x1[-1]
            x2 = self.adjust_depth(inputs=x2, name_prefix='{0}_adjustDepth'.format(name_prefix),
                                   current_filters=currentFilters)
        if shape_x1[-1] < shape_x2[-1]:
            currentFilters = shape_x2[-1]
            x1 = self.adjust_depth(inputs=x1, name_prefix='{0}_adjustDepth'.format(name_prefix),
                                   current_filters=currentFilters)

        x = Add(name='{0}_add'.format(name_prefix))([x1, x2])
        return x

    # 最后一层，全局池化+全连接+softmax激活
    def classification_layer(self, inputs, name_prefix, classes):
        x = GlobalAveragePooling2D(
            name="{0}_gap2d_".format(name_prefix))(inputs)
        x = Dense(
            classes,
            kernel_initializer=initializers.he_normal(),
            kernel_regularizer=regularizers.l2(),
            name="{0}_dense_".format(name_prefix))(x)
        x = Activation("softmax", name="{0}_softmax_".format(name_prefix))(x)
        return x


def dict_argmax(d):
    """
    find the max integer element's corresponding key value in the dict
    :param d: the dic object on which to perform argmax operation
    :return: the max integer element's corresponding key
    """
    assert isinstance(d, dict)
    max_value = 0
    max_key = list(d.keys())[0]
    for key in d.keys():
        if d[key] > max_value:
            max_value = d[key]
            max_key = key
    if max_value == 0:  # still 0, random chose
        max_key = np.random.choice(list(d.keys()))
    return max_key


def random_crop_image(image):
    from scipy import misc
    height, width = image.shape[:2]
    random_array = np.random.random(size=4)
    w = int((width * 0.5) * (1 + random_array[0] * 0.5))
    h = int((height * 0.5) * (1 + random_array[1] * 0.5))
    x = int(random_array[2] * (width - w))
    y = int(random_array[3] * (height - h))

    image_crop = image[y:h + y, x:w + x, 0:3]
    image_crop = misc.imresize(image_crop, image.shape)
    return image_crop


def draw_accuracy_each_iter(accuracyList, saveDir=None):
    """
    draw the accuracy of each iteration
    :param accuracyList:
    :param saveDir:
    :return:
    """
    import matplotlib.pyplot as plt
    assert isinstance(accuracyList, list)
    accuracyList.insert(0, 0.1)  # the accuracy in the first iter is random guess accuracy
    x = list(range(0, len(accuracyList)))
    plt.plot(x, accuracyList)
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.title("Q-learning process")
    if saveDir is not None:
        plt.savefig(os.path.join(saveDir, 'Q_learning_Iterations.png'))
    plt.show()  # 注意,这里savefig一定要写在前面,因为plt.show会创建新的空白图片


def compute_FLOP(nl, nL, sl, ml):
    """
    compute the flop of the model (operation complex)
    :param nl: channel(filter) num of the input
    :param nL: filter num
    :param sl: kernel size
    :param ml: output spacial size(length)
    :return: the flop estimation
    """
    return nl * nL * sl * sl * ml * ml


def get_refined_accuracy(early_accuracy, S, verbose=False, u=1, r=8):
    """
    using the current model and the early_accuracy(accuracy in a early stopped model)
    to predict the real accuracy(accuracy get after proper training epochs)

    following the paper "convolutional neural networks at constrained time cost"

    using FLOP and density of the model to do the prediction
    """

    # to compute the flop, there is a baseline needed
    # assume that the spacial size inner a block is 56(or any >1 number is ok)
    # assume that the filter num is 32
    flop_baseline = compute_FLOP(32, 32, 7, 56) + compute_FLOP(32, 32, 5, 56) + (
            compute_FLOP(32, 32, 3, 56) + compute_FLOP(32, 32, 3, 56)*2)
    flop = 0
    dot_num = len(S)
    edge_num = 0
    for s in S:
        Index, Type, KernelSize, Pred1, Pred2 = s.split('_')
        if Type in (5, 6):
            edge_num += 2
        else:
            edge_num += 1

        if Type == 1:  # PCC_conv
            flop += compute_FLOP(32, 32, KernelSize, 56)
    density = edge_num / dot_num
    flop /= flop_baseline

    estimate_accuracy = early_accuracy - u * np.log(flop / 5) - r * np.log(density / 5)

    if verbose:
        print("the estimate density of the model: "+str(density))
        print("the estimate flops of the model: "+str(flop))
        print("the estimate accuracy of the model is "+str(estimate_accuracy))

    return estimate_accuracy


def adjust_epsilon_schedule(epsilon_schedule, time_per_iter, time_limit):
    """
    按照时间限制等比例缩减每个epsilon对应的iteration的数量
    :param epsilon_schedule: 原来的epsilon_schedule
    :param time_per_iter: 一个迭代的时间
    :param time_limit: 整个nas的时间限制
    :return:
    """
    iteration_num = 0  # the number of iteration during the whole training process
    for iter_num_i in epsilon_schedule:
        iteration_num += iter_num_i  # the iteration num for a certain epsilon
    time_cost = iteration_num * time_per_iter
    reduce_factor = time_limit / time_cost  #
    if reduce_factor > 1:
        return epsilon_schedule
    else:
        return epsilon_schedule * reduce_factor