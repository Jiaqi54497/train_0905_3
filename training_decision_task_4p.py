#Training Simple Decision Making task
import numpy as np
import argparse
import sys
sys.path.insert(0, '../../../') #This line adds '../..' to the path so we can import the net_framework python file
from RNN_model_GRAD import *
import tensorflow as tf
from tensorflow import keras
import json
from tqdm import tqdm
import os
import pickle

def main(args):

    num_iters = args.num_iters
    num_nodes = args.num_nodes
    num_networks = args.num_networks

    for network_number in range(num_networks):
        #Defining Network
        time_constant = 100 #ms
        timestep = 10 #ms
        noise_strength = .01
        num_inputs = 16

        connectivity_matrix = np.ones((num_nodes, num_nodes))
        weight_matrix = np.random.normal(0, 1.2/np.sqrt(num_nodes), (num_nodes, num_nodes))
        for i in range(num_nodes):
            weight_matrix[i,i] = 0
            connectivity_matrix[i,i] = 0
        weight_matrix = tf.Variable(weight_matrix)
        connectivity_matrix = tf.constant(connectivity_matrix)

        noise_weights = 1 * np.ones(num_nodes)
        bias_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)
        input1_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)/2
        input2_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)/2
        input3_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)/2
        input4_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)/2
        input5_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)/2
        input6_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)/2
        input7_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)/2
        input8_weights = np.random.normal(0, 1/np.sqrt(num_inputs), num_nodes)/2


        input_weight_matrix = tf.constant(np.vstack((bias_weights, noise_weights, input1_weights, input2_weights, input3_weights, input4_weights, input5_weights, input6_weights, input7_weights, input8_weights)))

        def input1(time):
            #No input for now
            return 0
        def input2(time):
            return 0
        def input3(time):
            return 0
        def input4(time):
            return 0
        def input5(time):
            return 0
        def input6(time):
            return 0
        def input7(time):
            return 0
        def input8(time):
            return 0

        def bias(time):
            return 1
        def noise(time):
            return np.sqrt(2 * time_constant/timestep) * noise_strength * np.random.normal(0, 1)


        input_funcs = [bias, noise, input1, input2, input3, input4, input5, input6, input7, input8]
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1Change the number of outputs here.
        init_activations = tf.constant(np.zeros((num_nodes, 4)))
        print('init_activations:',init_activations.shape)
        output_weight_matrix = tf.constant(np.random.uniform(0, 1/np.sqrt(num_nodes), (4, num_nodes)))

        network = RNN(weight_matrix, connectivity_matrix, init_activations, output_weight_matrix, time_constant = time_constant,
                     timestep = timestep, activation_func = keras.activations.relu, output_nonlinearity = lambda x : x)

        #Training Network
        net_weight_history = {}

        time = 15000 #ms
        def gen_functions():
            switch_time = int(np.random.normal(time/2, time/10))

            #val1 = np.random.uniform(0, 2)
            #val2 = np.random.uniform(0, 2)
            #val3 = np.random.uniform(0, 2)
            #val4 = np.random.uniform(0, 2)
            val1 = 0.7
            val2 = 1.5
            val3 = 0.75
            val4 = 1.45
            val5 = 0.75
            val6 = 0.45
            val7 = 1.25
            val8 = 0.90
            val9 = 1.5
            val10= 0.6
            val11= 1.25
            val12= 0.35
            val13= 0.15
            val14= 1.66
            val15= 0.85
            val16= 1.15


            def input1(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return val1 + np.random.normal(0, .01)
                else:
                    return val9 + np.random.normal(0, .01)

            def input2(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return val2 + np.random.normal(0, .01)
                else:
                    return val10 + np.random.normal(0, .01)

            def input3(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return val3 + np.random.normal(0, .01)
                else:
                    return val11 + np.random.normal(0, .01)

            def input4(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return val4 + np.random.normal(0, .01)
                else:
                    return val12 + np.random.normal(0, .01)

            def input5(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return val5 + np.random.normal(0, .01)
                else:
                    return val13 + np.random.normal(0, .01)

            def input6(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return val6 + np.random.normal(0, .01)
                else:
                    return val14 + np.random.normal(0, .01)

            def input7(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return val7 + np.random.normal(0, .01)
                else:
                    return val15 + np.random.normal(0, .01)

            def input8(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return val8 + np.random.normal(0, .01)
                else:
                    return val16 + np.random.normal(0, .01)

            def target_func_1(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return 0.5 * (val1 > val5) + .8 * (val5 > val1)
                else:
                    return 0.5 * (val9 > val13) + .8 * (val13 > val9)

            def target_func_2(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return 0.5 * (val2 > val6) + .8 * (val6 > val2)
                else:
                    return 0.5 * (val10 > val14) + .8 * (val14 > val10)

            def target_func_3(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return 0.5 * (val3 > val7) + .8 * (val7 > val3)
                else:
                    return 0.5 * (val11 > val15) + .8 * (val15 > val11)

            def target_func_4(time):
                #running for 15 seconds = 15000ms
                if time < switch_time:
                    return 0.5 * (val4 > val8) + .8 * (val8 > val4)
                else:
                    return 0.5 * (val12 > val16) + .8 * (val16 > val12)

            def error_mask_func(time):
                #Makes loss automatically 0 during switch for 150 ms.
                #Also used in next training section.
                if time < 100:
                    return 0
                if time < switch_time + 100 and time > switch_time - 50:
                    return 0
                else:
                    return 1
            return input1, input2, input3, input4, input5, input6, input7, input8, target_func_1, target_func_2, target_func_3, target_func_4, error_mask_func

        targets = []
        inputs = []
        error_masks = []
        print('Preprocessing...', flush = True)
        for iter in tqdm(range(num_iters * 5), leave = True, position = 0):
            input1, input2, input3, input4, input5, input6, input7, input8, target_func_1, target_func_2, target_func_3, target_func_4, error_mask_func = gen_functions()
            targets.append(network.convert(time, [target_func_1,target_func_2,target_func_3,target_func_4]))
#            targets.append(network.convert(time, [target_func_2]))
#            targets.append(network.convert(time, [target_func_3]))
#            targets.append(network.convert(time, [target_func_4]))
            input_funcs[2] = input1
            input_funcs[3] = input2
            input_funcs[4] = input3
            input_funcs[5] = input4
            input_funcs[6] = input5
            input_funcs[7] = input6
            input_funcs[8] = input7
            input_funcs[9] = input8

            inputs.append(network.convert(time, input_funcs))
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            error_masks.append(network.convert(time, [error_mask_func,error_mask_func,error_mask_func,error_mask_func]))
        print('Training...', flush = True)
        print('Targets:',len(targets))
        print('Inputs:',len(inputs))

        weight_history, losses = network.train(num_iters, targets, time, num_trials = 10, inputs = inputs,
                      input_weight_matrix = input_weight_matrix, learning_rate = .001, error_mask = error_masks, save = 20)

        net_weight_history['trained weights'] = np.asarray(weight_history).tolist()

        net_weight_history['bias'] = bias_weights.tolist()
        net_weight_history['noise weights'] = noise_weights.tolist()
        net_weight_history['input1 weights'] = input1_weights.tolist()
        net_weight_history['input2 weights'] = input2_weights.tolist()
        net_weight_history['input3 weights'] = input3_weights.tolist()
        net_weight_history['input4 weights'] = input4_weights.tolist()
        net_weight_history['input5 weights'] = input5_weights.tolist()
        net_weight_history['input6 weights'] = input6_weights.tolist()
        net_weight_history['input7 weights'] = input7_weights.tolist()
        net_weight_history['input8 weights'] = input8_weights.tolist()
        net_weight_history['connectivity matrix'] = np.asarray(connectivity_matrix).tolist()
        net_weight_history['output weights'] = np.asarray(output_weight_matrix).tolist()

        if not os.path.isdir(args.savedir):
            os.mkdir(args.savedir)

        network_params = {}
        network_params['n_nodes'] = num_nodes
        network_params['time_constant'] = time_constant
        network_params['timestep'] = timestep
        network_params['noise_strength'] = noise_strength
        network_params['num_input'] = num_inputs

        with open('%s/network_params.dat' % args.savedir, 'wb') as f:
            f.write(pickle.dumps(network_params))

        if not os.path.isdir('%s/weight_history_' % args.savedir):
            os.mkdir('%s/weight_history_' % args.savedir)

        with open('%s/weight_history_' % args.savedir + str(network_number)+'.json', 'w') as f:
            json.dump(net_weight_history, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('savedir')
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--num_nodes', type=int, default=256)
    parser.add_argument('--num_networks', type=int, default=1)

    args = parser.parse_args()
    main(args)
