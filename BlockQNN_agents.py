import numpy as np
from BlockQNN_tools import dict_argmax, QNN_layers, random_crop_image
from keras.layers import Conv2D, MaxPooling2D
from keras import Model
from keras import optimizers
from keras.datasets import cifar10
import keras
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as bk
from keras.callbacks import LearningRateScheduler
import os


class MasterAgent:
    """
    each operation is a dic item like:
    'Index_Type_Kernel size_Pred1_Pred2'
    """

    def __init__(self, T, q_lr=0.01, gamma=1.0):  # T is the total time steps
        self.T = T
        self.q_lr = q_lr  # the learning rate for Bellman equation
        self.Q_table = self.initiate_q_table()
        self.gamma = gamma

    def initiate_q_table(self):
        q_table = {}
        """
        use dict to store the Q_table
        Type == 1 : PCC_conv
        Type == 2 : max_pooling
        Type == 3 : Average_pooling
        Type == 4 : Identity
        Type == 5 : Add
        Type == 6 : Concat
        Type == 7 : Terminal
        """
        q_table['0_input'] = {}
        for index in range(1, self.T + 1):  # the right side is not included
            # initiate states
            for TYPE in range(1, 8):  # there are 7 kinds of Type
                if TYPE == 1:  # convolution
                    for kernel in (1, 3, 5):
                        for pred1 in range(0, index):
                            q_table["{0}_{1}_{2}_{3}_{4}".format(
                                index, TYPE, kernel, pred1, 0)] = {}
                elif TYPE in (2, 3):
                    for kernel in (1, 3):
                        for pred1 in range(0, index):
                            q_table["{0}_{1}_{2}_{3}_{4}".format(
                                index, TYPE, kernel, pred1, 0)] = {}
                elif TYPE == 4:
                    for pred1 in range(0, index):
                        q_table["{0}_{1}_{2}_{3}_{4}".format(
                            index, TYPE, 0, pred1, 0)] = {}
                elif TYPE in (5, 6):
                    if index < 2:  # add and concat operation need at least two inputs(input layer can be included)
                        continue
                    for pred1 in range(0, index):
                        for pred2 in range(0, index):
                            if pred1 == pred2:
                                continue
                            q_table["{0}_{1}_{2}_{3}_{4}".format(
                                index, TYPE, 0, pred1, pred2)] = {}
                elif TYPE == 7:
                    if index == 1:  # output = input, meaningless
                        continue
                    q_table["{0}_{1}_{2}_{3}_{4}".format(
                        index, TYPE, 0, 0, 0)] = {}

        # initiate actions
        # original reward is 0.5 as random guessing accuracy
        for state in q_table.keys():
            index = int(state.split('_')[0]) + 1  # the state's action can chose the state itself as predecessor
            for TYPE in range(1, 8):  # there are 7 kinds of Type
                if TYPE == 1:  # convolution
                    for kernel in (1, 3, 5):
                        for pred1 in range(0, index):
                            q_table[state]["{0}_{1}_{2}_{3}".format(
                                TYPE, kernel, pred1, 0)] = 0.5
                elif TYPE in (2, 3):
                    for kernel in (1, 3):
                        for pred1 in range(0, index):
                            q_table[state]["{0}_{1}_{2}_{3}".format(
                                TYPE, kernel, pred1, 0)] = 0.5
                elif TYPE == 4:
                    for pred1 in range(0, index):
                        q_table[state]["{0}_{1}_{2}_{3}".format(
                            TYPE, 0, pred1, 0)] = 0.5
                elif TYPE in (5, 6):
                    if state == '0_input':  # add and concat operation need at least two inputs
                        continue
                    for pred1 in range(0, index):
                        for pred2 in range(0, index):
                            if pred1 == pred2:
                                continue
                            q_table[state]["{0}_{1}_{2}_{3}".format(
                                TYPE, 0, pred1, pred2)] = 0.5
                elif TYPE == 7:
                    if state == '0_input':  # output = input, meaningless
                        continue
                    q_table[state]["{0}_{1}_{2}_{3}".format(
                        TYPE, 0, 0, 0)] = 0.5
        return q_table

    def sample_new_network(self, epsilon):
        """
        based on the algorithm posted in the metaQnn paper
        """
        # initialize S->state sequence;U->action sequence
        S = ['0_input']
        U = []
        index = 1

        # not the terminate layer and not surpass the max index(can be infinite)
        while index <= self.T:
            a = np.random.uniform(0, 1)
            if a > epsilon:  # exploration
                u = dict_argmax(self.Q_table[S[-1]])
                new_state = str(index) + '_' + u
            else:  # exploitation
                u = np.random.choice(list(self.Q_table[S[-1]].keys()))
                new_state = str(index) + '_' + u
            U.append(u)
            if u != '7_0_0_0':  # u != terminate
                S.append(new_state)
            else:
                return S, U
            index += 1
        U.append('7_0_0_0')
        return S, U

    def update_q_values(self, S, U, accuracy, gamma=1.0):
        """
        based on the algorithm posted in the metaQnn paper
        :param gamma: the discount factor which measures the importance of future rewards
        :param S: state sequence
        :param U: action sequence
        :param accuracy: the model accuracy on the validation set
        :return: None
        """
        self.Q_table[S[-1]][U[-1]] = (1 - self.q_lr) * self.Q_table[S[-1]][U[-1]] \
                                     + self.q_lr * accuracy

        rt = accuracy / self.T

        # find the max action reward for the next step
        i = len(S) - 2
        while i >= 0:
            max_action_reward = 0
            for action in list(self.Q_table[S[i + 1]].keys()):
                if self.Q_table[S[i + 1]][action] > max_action_reward:
                    max_action_reward = self.Q_table[S[i + 1]][action]

            self.Q_table[S[i]][U[i]] = (1 - self.q_lr) * self.Q_table[S[i]][U[i]] \
                                       + self.q_lr * (rt + gamma * max_action_reward)
            i -= 1


class ControllerAgent:
    def __init__(self, N, T, input_shape):
        self.qnn_layers = QNN_layers()
        self.input_layer = self.qnn_layers.input_layer(input_shape=input_shape, name_prefix='input')
        self.N = N
        self.T = T

    @staticmethod
    def generate_layer_name(block_index, index, predecessors):
        return "inBlock{0}_{1}_{2}".format(block_index, index, predecessors)

    def generate_layer(self, s, layer_outputs, current_filters, follow, block_index):
        Index, Type, KernelSize, Pred1, Pred2 = s.split('_')
        KernelSize = int(KernelSize)
        follow[int(Pred1)] = 1
        follow[int(Pred2)] = 1

        if Type == "1":
            name_prefix = self.generate_layer_name(block_index, Index,
                                                   'pred_{0}_{1}'.format(Pred1, Pred2))
            return self.qnn_layers.PCC_conv(
                inputs=layer_outputs[Pred1], name_prefix=name_prefix,
                kernel_size=(KernelSize, KernelSize), current_filters=current_filters)

        elif Type == "2":
            name_prefix = self.generate_layer_name(block_index, Index,
                                                   'pred_{0}_{1}'.format(Pred1, Pred2))
            return self.qnn_layers.Max_pooling(
                inputs=layer_outputs[Pred1], name_prefix=name_prefix, kernel_size=(KernelSize, KernelSize))

        elif Type == "3":
            name_prefix = self.generate_layer_name(block_index, Index,
                                                   'pred_{0}_{1}'.format(Pred1, Pred2))
            return self.qnn_layers.Average_pooling(
                inputs=layer_outputs[Pred1], name_prefix=name_prefix, kernel_size=(KernelSize, KernelSize))
        elif Type == "4":
            name_prefix = self.generate_layer_name(block_index, Index,
                                                   'pred_{0}_{1}'.format(Pred1, Pred2))
            return self.qnn_layers.identity(inputs=layer_outputs[Pred1], name_prefix=name_prefix)
        elif Type == "5":
            name_prefix = self.generate_layer_name(block_index, Index,
                                                   'pred_{0}_{1}'.format(Pred1, Pred2))
            return self.qnn_layers.elemental_add(name_prefix=name_prefix, x1=layer_outputs[Pred1],
                                                 x2=layer_outputs[Pred2])
        elif Type == "6":
            name_prefix = self.generate_layer_name(block_index, Index,
                                                   'pred_{0}_{1}'.format(Pred1, Pred2))
            return self.qnn_layers.concat(name_prefix=name_prefix,
                                          input_list=[layer_outputs[Pred1], layer_outputs[Pred2]],
                                          current_filters=current_filters)

    def generate_block(self, S, current_filters, inputs, block_index):
        """
        generate a actual network based on the state sequence
        :param block_index: the block's index in the whole network
        :param inputs: the input for the block
        :param current_filters: the filter number for the certain level of block
        :param S: the state sequence
        :return: a new network model
        """
        print(S)

        # initiate the subsequent array(whether a certain layer's output is feed into another layer)
        follow = []
        for i in range(0, self.T + 1):
            follow.append(0)
        follow[0] = 1  # input layer must feed as input of at least one other layers

        # initiate a dict to store the output of each layer
        layer_outputs = {'0': inputs}65tg

        # load the state sequence and transfer into real network
        for s in S:
            if s.split('_')[0] == str(0):
                continue
            index = s.split('_')[0]
            output = self.generate_layer(s, layer_outputs, current_filters, follow, block_index=block_index)
            layer_outputs[index] = output

        # some layers that never feed as other layers' inputs are concatenated together to generate the final output
        concat_layers = []
        concat_layers_string = ""  # use as name prefix
        for i in range(1, len(S)):
            if follow[i] == 0:
                concat_layers_string += str(i) + '_'
                concat_layers.append(layer_outputs[str(i)])
        if len(concat_layers) == 1:
            return concat_layers[0]

        name_prefix = self.generate_layer_name(block_index, 'block_output', concat_layers_string)
        block_output = self.qnn_layers.concat(name_prefix=name_prefix, input_list=concat_layers,
                                              current_filters=current_filters)
        return block_output

    def generate_network(self, S, isCifar10=True):
        if isCifar10:
            inputs = self.input_layer
            x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', name='conv')(inputs)
            block_index = 0
            current_filters = 32
            for i in range(0, self.N):
                block_index += 1
                x = self.generate_block(S=S, current_filters=current_filters, inputs=x, block_index=block_index)
            x = MaxPooling2D()(x)
            current_filters = 64
            for i in range(0, self.N):
                block_index += 1
                x = self.generate_block(S=S, current_filters=current_filters, inputs=x, block_index=block_index)
            x = MaxPooling2D()(x)
            current_filters = 128
            for i in range(0, self.N):
                block_index += 1
                x = self.generate_block(S=S, current_filters=current_filters, inputs=x, block_index=block_index)
            outputs = self.qnn_layers.classification_layer(
                inputs=x, name_prefix='final_classification', classes=10)

            return Model(inputs=inputs, outputs=outputs)


class ComputeAgent:
    def __init__(self, network_model, model_name, epoch=12, minibatch_size=256,
                 opt=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                 ):
        self.network_model = network_model
        self.epoch = epoch
        self.minibatch_size = minibatch_size
        self.opt = opt
        self.model_name = model_name

    @staticmethod
    def generate_cifar10data():
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # implement data augmentation for cifar10
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            preprocessing_function=random_crop_image
        )

        datagen.fit(x_train)
        return x_train, x_test, y_train, y_test, datagen

    def train_network(self, isCifar10=True, save_dir='saved_models'):
        assert isinstance(self.network_model, Model)
        self.network_model.compile(optimizer=self.opt,
                                   loss='categorical_crossentropy',
                                   metrics=['accuracy'])
        if isCifar10:
            x_train, x_test, y_train, y_test, datagen = self.generate_cifar10data()
            datagen.fit(x_train)

            # define the lr_decay_schedule
            def lr_decay_schedule(epoch):
                lr = bk.get_value(self.network_model.optimizer.lr)
                # 每隔5个epoch，学习率减小为原来的1/5
                if epoch % 5 == 0 and epoch != 0:
                    lr *= 0.2
                return lr

            lr_decay = LearningRateScheduler(schedule=lr_decay_schedule, verbose=1)

            self.network_model.fit_generator(
                datagen.flow(x_train, y_train, batch_size=self.minibatch_size),
                steps_per_epoch=x_train.shape[0] / self.minibatch_size,
                epochs=self.epoch, validation_data=(x_test, y_test), workers=4,
                callbacks=[lr_decay])

            # save the trained model
            save_dir = os.path.join(os.getcwd(), save_dir)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, self.model_name)
            self.network_model.save(model_path)
            print('Saved trained model at %s' % model_path)

            # evaluate the trained model and get it's accuracy
            (loss, accuracy) = self.network_model.evaluate(x=x_test, y=y_test)
            return accuracy

    def train_best_network(self, isCifar10=True, save_dir='saved_best_models'):
        assert isinstance(self.network_model, Model)
        self.network_model.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0005),
                                   loss='categorical_crossentropy',
                                   metrics=['accuracy'])
        if isCifar10:
            x_train, x_test, y_train, y_test, datagen = self.generate_cifar10data()
            datagen.fit(x_train)

            # define the lr_decay_schedule
            def lr_decay_schedule(epoch):
                lr = bk.get_value(self.network_model.optimizer.lr)
                # 每隔30个epoch，学习率减小为原来的1/10
                if epoch == 150:
                    lr = 0.01
                if epoch == 225:
                    lr = 0.001
                return lr

            lr_decay = LearningRateScheduler(schedule=lr_decay_schedule, verbose=1)

            self.network_model.fit_generator(
                datagen.flow(x_train, y_train, batch_size=128),
                steps_per_epoch=x_train.shape[0] / self.minibatch_size,
                epochs=300, validation_data=(x_test, y_test), workers=4,
                callbacks=[lr_decay])

            # save the trained model
            save_dir = os.path.join(os.getcwd(), save_dir)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, self.model_name)
            self.network_model.save(model_path)
            print('Saved trained model at %s' % model_path)

            # evaluate the trained model and get it's accuracy
            (loss, accuracy) = self.network_model.evaluate(x=x_test, y=y_test)
            return accuracy
