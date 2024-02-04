import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import time

from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer

from utility import * 

import logging

class FedMD():
    def __init__(self, parties, original_public_dataset, 
                 private_data,  
                 private_test_data,
                 N_rounds, N_alignment, 
                 N_logits_matching_round, logits_matching_batchsize, 
                 N_private_training_round, private_training_batchsize,
                 aug = False, compress = False, select = False):
        
        self.N_parties = len(parties)
        self.original_public_dataset = original_public_dataset
        N_alignment_per_class = N_alignment // len(self.original_public_dataset)
        self.public_dataset = [self.original_public_dataset[i][:N_alignment_per_class] for i in range(len(self.original_public_dataset))]
        self.public_classes = np.arange(len(self.public_dataset))
        self.private_data = private_data
        self.private_test_data = private_test_data
        
        self.N_rounds = N_rounds
        self.N_alignment = N_alignment
        self.aug = aug
        self.compress = compress
        self.select = select 
        self.heatmaps = np.zeros((int(self.N_rounds), self.N_parties, len(self.public_classes)))

        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize
        
        self.collaborative_parties = []
        self.init_result = []

        self.logger = logging.getLogger("parent")
        self.logger.setLevel(logging.INFO)

        self.rounds_time = []

        
        
        # print("start model initialization: ")
        for i in range(self.N_parties):
            print("model ", i)
            self.logger.info("model {0}".format(i))
            model_A_twin = None
            model_A_twin = clone_model(parties[i])
            model_A_twin.set_weights(parties[i].get_weights())
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-4), 
                                 loss = "categorical_crossentropy",
                                 metrics = ["accuracy"])
            
            model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")
            
            self.collaborative_parties.append({"model_logits": model_A, 
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights()})
            
    def collaborative_training(self):  
        client_time = 0 

        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        
        rounds_start_time = time.time()
        last_time_stamp = time.perf_counter_ns()
        while True:
            
            # 1. augmenting public data 
            #print("start with: augmenting public dataset")
            #print("ends with round ")

            # 2. calculate local logits 
            # 3. upload local logits to the server
            # 4. server aggregate local logits
            # 5. Clients download global logits 
            # print(start with update logits)
            # print(ends with local logits shape ) / num_clients

            # 6. clients perform knowledge distillation training 
            # print(starts with selected labels shape)
            # print(ends with starting training with private data)

            # 7. clients perform private training
            # print(starts with starting training with private data)
            # print(ends with done private training)

            # 8. clients test performance on test sets

            if r > 0 : 
                self.rounds_time.append(time.time() - rounds_start_time)
                rounds_start_time = time.time()
            
            X = np.concatenate(self.public_dataset)
            y = np.repeat(range(len(self.public_dataset)), len(self.public_dataset[0]))

            t1 = time.time() 
            if not self.aug : 
                # At beginning of each round, generate new alignment dataset
                alignment_data = generate_alignment_data(X, y, N_alignment= "all") 
                
            else : 
                print("augmenting public dataset ... ")
                self.logger.info("augmenting public dataset ... ")
                alpha = np.random.randint(1, 1_000_000)
                beta = np.random.randint(1, 1000)
                lambdaa = np.random.beta(alpha, alpha)
                
                np.random.seed(beta) 
                index = np.random.permutation(len(X))  
                new_public_dataset_x = lambdaa * X + (1 - lambdaa) * X[index]
                new_public_dataset_y = y[index]
                # new_public_dataset_y = lambdaa * self.public_dataset["y"] + (1 - lambdaa) * self.public_dataset["y"][index]

                # At beginning of each round, generate new alignment dataset
                alignment_data = generate_alignment_data(new_public_dataset_x, 
                                                        new_public_dataset_y,
                                                        N_alignment= "all")
            
            alignment_data_preparation_time = time.perf_counter_ns() - last_time_stamp
            last_time_stamp = time.perf_counter_ns()
            self.logger.info("alignment_data_preparation_time: {0}".format(alignment_data_preparation_time))

            print("round ", r)
            print("generated {} alignment data in {} seconds:".format(len(alignment_data), alignment_data_preparation_time / 1e-9))
            print("alignment_data shape x:{}  Y:{}".format(alignment_data['X'].shape, alignment_data['y'].shape))
            self.logger.info("round {0}".format(r))
            
            print("update logits ... ")
            self.logger.info("update logits ... ")
            # update logits
            # print("aug:{0}, compress:{1}, N_alignment:{2}".format(self.aug, self.compress, self.N_alignment))
            # print("collaborative parties", len(self.collaborative_parties))
            # print("size of alignment data {0}, length: {1}".format(size_of(alignment_data['y']), len(alignment_data["y"])))
            local_logits = []
            for d in self.collaborative_parties:
                d["model_logits"].set_weights(d["model_weights"])
                client_logits = [] 
                for c in self.public_classes : 
                    X = alignment_data["X"][alignment_data["y"] == c]
                    client_logits.append(d["model_logits"].predict(X, verbose = 0))
                local_logits.append(np.array(client_logits)) 
            ll = np.array(local_logits)

            soft_labels_generation_time = (time.perf_counter_ns() - last_time_stamp ) / len(self.collaborative_parties)
            last_time_stamp = time.perf_counter_ns()
            self.logger.info("soft_labels_generation_time: {0}".format(soft_labels_generation_time))

            t2 = time.time() 
            # print("model summary:", d['model_logits'].summary())
            # print("GT shape:", alignment_data["y"].shape)
            
            global_logits = []
            for i in range(len(self.public_dataset)): 
                logits = aggregate(ll[:, i], self.compress) 
                global_logits.append(logits)
            # print("size of local soft labels:{0}, size of global soft labels:{1}".format(size_of(local_logits[0]), size_of(logits)))
            # print("length of local soft labels:{0}, length of global soft labels:{1}".format(len(local_logits[0]), len(logits)))
            # print("type of local soft labels:{0}, type of global soft labels:{1}".format(local_logits[0].dtype, logits.dtype))
            # return 

            soft_labels_aggregation_time = time.perf_counter_ns() - last_time_stamp
            last_time_stamp = time.perf_counter_ns()
            self.logger.info("soft_labels_aggregation_time: {0}".format(soft_labels_aggregation_time))

            ll = np.array(local_logits) 
            gl = np.array(global_logits)

            print("local logits shape: ", ll.shape)       
            print("global logits shape: ", gl.shape)
            self.logger.info("local logits shape: {0}".format(ll.shape))
            self.logger.info("global logits shape: {0}".format(gl.shape))

            diff_l = np.power(ll - gl, 2)
            mean_diff_l = np.mean(diff_l, axis = 0)
            print("diff_l: ", diff_l.shape)
            print("mean_diff_l: ", mean_diff_l.shape)
            
            client_distance_map = np.zeros((self.N_parties, len(self.public_classes))) 
            for i in self.public_classes : 
                for c in range(self.N_parties):
                    print("i:{}  c:{}".format(i, c))

                    dist = np.mean(diff_l[c, i])
                    client_distance_map[c, i] = dist
            
            classes_to_clients = []
            dist_threshold = 0.0 if not self.select else 0.5 # 0 means no selection
            for i in self.public_classes : 
                norm_dist = (client_distance_map[:, i] - np.min(client_distance_map[:, i])) / (np.max(client_distance_map[:, i]) )
                weak_clients = np.where(norm_dist >= dist_threshold)[0]
                strong_clients = np.where(norm_dist < dist_threshold)[0]
                classes_to_clients.append({"weak": weak_clients, "strong": strong_clients})
            clients_to_classes = {i: [] for i in range(self.N_parties)}
            for i, item in enumerate(classes_to_clients):
                for c in item["weak"]:
                    clients_to_classes[c].append(i)

            selected_data = [] 
            selected_labels = []
            for i in range(self.N_parties):
                selected_data.append([])
                selected_labels.append([])
                for c in clients_to_classes[i]:
                    X = alignment_data["X"][alignment_data["y"] == c]
                    selected_data[i].append(X)
                    selected_labels[i].append(gl[ c])
                if len(selected_data[i]) : 
                    selected_data[i] = np.concatenate(selected_data[i], axis = 0)
                    selected_labels[i] = np.concatenate(selected_labels[i], axis = 0)
                else :
                    selected_data[i] = np.array([])
                    selected_labels[i] = np.array([])

            # normalize the distance map
            # client_distance_map -= np.min(client_distance_map)
            # client_distance_map = client_distance_map / np.max(client_distance_map)
            # self.heatmaps[r] = client_distance_map
            

                
            
            print("ratio of data saved by selection")
            for i in range(self.N_parties):
                print(len(selected_data[i]) / len(alignment_data["X"]))
            print()

            self.logger.info("ratio of data saved by selection")
            for i in range(self.N_parties):
                self.logger.info("{0}".format(len(selected_data[i]) / len(alignment_data["X"])))
            self.logger.info("")

            soft_labels_selection_time = time.perf_counter_ns() - last_time_stamp
            last_time_stamp = time.perf_counter_ns()
            self.logger.info("soft_labels_selection_time: {0}".format(soft_labels_selection_time))

            t3 = time.time() 
            print("updates models ...")
            avg_kd_training_time, avg_private_training_time = [], []
            for index, d in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))
                self.logger.info("model {0} starting alignment with public logits... ".format(index))
                
                

                weights_to_use = d["model_weights"]
                d["model_logits"].set_weights(weights_to_use)
                print("fitting")
                print("selected data shape: ", selected_data[index].shape)
                print("selected labels shape: ", selected_labels[index].shape)
                self.logger.info("selected data shape: {0}".format(selected_data[index].shape))
                self.logger.info("selected labels shape: {0}".format(selected_labels[index].shape))

                kd_training_start_time = time.perf_counter_ns()
                last_time_stamp = time.perf_counter_ns()

                # Knowledge distillation training
                if len(selected_data[index]) > 0:
                    d["model_logits"].fit(selected_data[index], selected_labels[index],
                                        batch_size = self.logits_matching_batchsize,  
                                        epochs = self.N_logits_matching_round, 
                                        shuffle=True, verbose = 0)
                    d["model_weights"] = d["model_logits"].get_weights()
                    print("model {0} done alignment".format(index))
                    self.logger.info("model {0} done alignment".format(index))
                
                kd_training_end_time = time.perf_counter_ns()
                last_time_stamp = time.perf_counter_ns()
                avg_kd_training_time.append(kd_training_end_time - kd_training_start_time)

                # Private training
                print("size of private_data:", self.private_data[index][0].shape, self.private_data[index][1].shape) 
                print("dtype:", self.private_data[index][0].dtype, self.private_data[index][1].dtype)
                print("model summary:", d['model_classifier'].summary())
                print("model {0} starting training with private data... ".format(index))
                self.logger.info("model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index][0], self.private_data[index][1],       
                                          batch_size = self.private_training_batchsize, 
                                          epochs = self.N_private_training_round, 
                                          shuffle=True, verbose = 0)

                private_training_end_time = time.perf_counter_ns()
                last_time_stamp = time.perf_counter_ns()
                avg_private_training_time.append(private_training_end_time - kd_training_end_time)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("model {0} done private training. \n".format(index))
                self.logger.info("model {0} done private training. \n".format(index))
            #END FOR LOOP
            t4 = time.time()

            self.logger.info("avg_kd_training_time: {0}".format(np.mean(avg_kd_training_time)))
            self.logger.info("avg_private_training_time: {0}".format(np.mean(avg_private_training_time)))
        
            # test performance
            print("test performance ... ")
            
            test_start_time = time.perf_counter_ns()

            for index, d in enumerate(self.collaborative_parties):
                predictions, labels = [], [] 
                for i, data in enumerate(self.private_test_data):
                    y_pred = d["model_classifier"].predict(data, verbose = 0).argmax(axis = 1)
                    label = np.repeat(i, len(y_pred))
                    predictions.append(y_pred) 
                    labels.append(label)
                predictions = np.concatenate(predictions, axis = 0)
                labels = np.concatenate(labels, axis = 0)
                collaboration_performance[index].append(np.mean(labels == predictions))
                
                print(collaboration_performance[index][-1])
                self.logger.info(collaboration_performance[index][-1])
                del y_pred
            
            test_end_time = time.perf_counter_ns()
            last_time_stamp = time.perf_counter_ns()
            testing_time = (test_end_time - test_start_time) / self.collaborative_parties
            self.logger.info("test_end_time: {0}".format(testing_time))
                
                
            r+= 1
            if r >= self.N_rounds:
                # if self.check_exit(collaboration_performance) : 
                break 
        
        client_time = (t4 - t3) + (t2 - t1) 
        # print("client time;", client_time)
        #END WHILE LOOP
        return collaboration_performance

        
    def check_exit(self, collaboration_performance) : 
        last_acc = np.mean([collaboration_performance[i][-1] for i in range(self.N_parties)])

        for i in range(-3, -1, -1) : 
            acc = np.mean([collaboration_performance[j][i] for j in range(self.N_parties)])
            if acc < last_acc : 
                return True
        
        return False





class FedAvg():
    def __init__(self, parties, private_data, 
                 private_test_data, N_rounds, N_private_training_round, private_training_batchsize):

        self.N_parties = len(parties)
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_rounds = N_rounds

        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize
        self.collaborative_parties = []

        self.logger = logging.getLogger("parent")
        self.logger.setLevel(logging.INFO)

        for i in range(self.N_parties):
            print("model ", i)
            self.logger.info("model {0}".format(i))
            model_clone = tf.keras.models.clone_model(parties[i])
            model_clone.set_weights(parties[i].get_weights())
            model_clone.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                                loss="categorical_crossentropy",
                                metrics=["accuracy"])
            
            self.collaborative_parties.append(model_clone) 


    def new_aggregate(self, weights) : 
        avg_weights = []
        for layer_id in range(len(weights[0]) ): 
            avg_layer = np.mean([weights[i][layer_id] for i in range(len(weights))], axis = 0)
            avg_weights.append(avg_layer)

        return avg_weights

    def aggregate_weights(self, models_weights):
        # Get the total number of layers in the model
        num_layers = len(models_weights[0])

        avg_weights = []

        # Iterate over each layer
        for layer in range(num_layers):
            # For each layer, get the average weight across all models
            layer_avg = np.mean([model[layer] for model in models_weights], axis=0)
            avg_weights.append(layer_avg)

        return avg_weights

    def collaborative_training(self):
        client_time = 0 
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0

        
        while True:
            print("round ", r)
            self.logger.info("round {0}".format(r))

            last_time_stamp = time.perf_counter_ns()
            t1 = time.time()
            # set all parties to the average weights
            for i, d in enumerate(self.collaborative_parties):
                d.fit(self.private_data[i][0], self.private_data[i][1],
                               batch_size=self.private_training_batchsize,
                               epochs= self.N_private_training_round,
                               shuffle=True, verbose=0)
                
            private_training_time = (time.perf_counter_ns() - last_time_stamp) / len(self.collaborative_parties)
            last_time_stamp = time.perf_counter_ns()
            self.logger.info("private_training_time: {0}".format(private_training_time))

            t2 = time.time()
            all_model_weights = [d.get_weights() for d in self.collaborative_parties]
            avg_weights = self.new_aggregate(all_model_weights)
            d.set_weights(avg_weights)

            model_aggregation_time = time.perf_counter_ns() - last_time_stamp
            last_time_stamp = time.perf_counter_ns()
            self.logger.info("model_aggregation_time: {0}".format(model_aggregation_time))
                

            for index, d in enumerate(self.collaborative_parties):
                predictions, labels = [], []
                for i, data in enumerate(self.private_test_data):
                    y_pred = d.predict(data, verbose = 0).argmax(axis = 1)
                    label = np.repeat(i, len(y_pred))
                    predictions.append(y_pred) 
                    labels.append(label)
                predictions = np.concatenate(predictions, axis = 0)
                labels = np.concatenate(labels, axis = 0)
                client_accuracy = np.mean(labels == predictions)
                collaboration_performance[index].append(client_accuracy)
                
                print("model {0} accuracy: {1}".format(index, client_accuracy))
                self.logger.info("model {0} accuracy: {1}".format(index, client_accuracy))
            
            testing_time = (time.perf_counter_ns() - last_time_stamp) / len(self.collaborative_parties)
            last_time_stamp = time.perf_counter_ns()
            self.logger.info("testing_time: {0}".format(testing_time))

            r += 1
            if r >= self.N_rounds :
                # if self.check_exit(collaboration_performance) : 
                break 
        
        client_time = t2 - t1
        # print("client time;", client_time)
                
        return collaboration_performance
    
        

        
    def check_exit(self, collaboration_performance) : 
        last_acc = np.mean([collaboration_performance[i][-1] for i in range(self.N_parties)])

        for i in range(-3, -1, -1) : 
            acc = np.mean([collaboration_performance[j][i] for j in range(self.N_parties)])
            if acc < last_acc : 
                return True
        
        return False

