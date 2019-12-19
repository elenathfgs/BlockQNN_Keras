#    (\~---.
#    /   (\-`-/)
#   (      ' ' )
#    \ (  \_Y_/\
#     ""\ \___//
#        `w   "    
# -implement by Elenath Feng

import time
from keras import optimizers
from BlockQNN_agents import MasterAgent, ControllerAgent, ComputeAgent
import numpy as np
from BlockQNN_tools import get_refined_accuracy, adjust_epsilon_schedule


class BlockQnn:
    def __init__(self,
                 N=4,
                 T=10,
                 sampleBlock_num=64,
                 batch_size=64,
                 max_run_time=43200,  # =12h, max running time of the program(seconds)
                 epsilon_schedule=None,
                 evaluate_best_model=True):
        """
        :param N: block repeat(stack) times
        :param T: maximum node num in a block
        """
        if epsilon_schedule is None:
            epsilon_schedule = [95, 7, 7, 7, 10, 10, 10, 10, 10, 12]
        self.epsilon_schedule = epsilon_schedule
        self.evaluate_best_model = evaluate_best_model
        self.masterAgent = MasterAgent(T=T)
        self.controllerAgent = ControllerAgent(N=N, T=T, input_shape=(32, 32, 3))
        self.sampleBlock_num = sampleBlock_num
        self.batch_size = batch_size
        self.max_run_time = max_run_time
        self.replay_memory = []

    def nas(self):
        epsilon = 1.1
        maxAccuracy = 0
        bestModel = None
        bestModel_description = None
        accuracy_each_iteration = []  # the list used to store the accuracy of each iteration
        startTime = time.time()

        for iteration_num in self.epsilon_schedule:
            epsilon -= 0.1  # start from epsilon == 1.0
            print('epsilon = {0}'.format(str(epsilon)))
            for i in range(0, iteration_num):
                begintime = time.time()  # begin time

                print('{0}th iteration'.format(i))

                S, U = self.masterAgent.sample_new_network(epsilon=epsilon)
                network_model = self.controllerAgent.generate_network(S=S)
                compute_agent = ComputeAgent(network_model=network_model, minibatch_size=self.batch_size,
                                             model_name="model{0}_epsilon_{1}.h5".
                                             format(str(iteration_num), str(epsilon)),
                                             # 这里声明一下优化器来去除优化器共享了学习率的bug
                                             opt=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999))
                origin_accuracy = compute_agent.train_network()
                accuracy = get_refined_accuracy(early_accuracy=origin_accuracy,
                                                S=S, verbose=True)
                if accuracy > maxAccuracy:
                    bestModel = network_model
                    bestModel_description = (S, U)
                accuracy_each_iteration.append(accuracy)

                self.replay_memory.append((S, U, accuracy))
                # experience replay
                for memory in range(0, np.min((len(self.replay_memory),
                                              self.sampleBlock_num))):
                    choiceIndex = np.random.choice(range(0, len(self.replay_memory)))
                    s_sample, u_sample, accuracy_sample = self.replay_memory[choiceIndex]
                    self.masterAgent.update_q_values(S=s_sample, U=u_sample,
                                                     accuracy=accuracy_sample)

                endtime = time.time()  # end time
                time_per_iter = endtime-begintime
                print("time for one iter= %d s" % str(time_per_iter))
                self.epsilon_schedule = adjust_epsilon_schedule(epsilon_schedule=self.epsilon_schedule,
                                                                time_per_iter=time_per_iter,
                                                                time_limit=self.max_run_time)

        compute_agent_best = ComputeAgent(network_model=bestModel, model_name="best_model",
                                          minibatch_size=self.batch_size)
        finishTime = time.time()
        print("time cost for the searching process is " + str(finishTime-startTime))

        if self.evaluate_best_model:
            best_accuracy = compute_agent_best.train_best_network()
            print('the final accuracy for best model is %f' % best_accuracy)
            print('the description for the best model: states:{0} actions:{1}'.
                  format(bestModel_description[0], bestModel_description[1]))
            # show and save the q-learning process
            from BlockQNN_tools import draw_accuracy_each_iter
            import os
            draw_accuracy_each_iter(accuracy_each_iteration,
                                    os.path.join(os.getcwd(), "trainingPictures"))

            finallyTime = time.time()
            print("time cost for the whole nas is" + str(finallyTime - startTime))
