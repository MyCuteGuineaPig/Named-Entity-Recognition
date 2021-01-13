#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:00:35 2021

@author: beckswu
"""

import trax 
from trax import layers as tl
import os 
import numpy as np

import random as rnd
from trax.supervised import training
import json
import contextlib


class NERClass:
    def __init__(self,):
        self.dir = os.path.dirname(os.path.abspath(__file__))
        
        self.model = None
        
        self.vocab, self.tag_map = self.get_vocab(self.dir+'/../data/words.txt', self.dir+'/../data/tags.txt')
        self.t_sentences, self.t_labels, self.t_size = self.get_params(self.vocab, self.tag_map, self.dir+'/../data/train/sentences.txt', self.dir+'/../data/train/labels.txt')
        self.v_sentences, self.v_labels, self.v_size = self.get_params(self.vocab, self.tag_map, self.dir+'/../data/val/sentences.txt', self.dir+'/../data/val/labels.txt')
        self.test_sentences, self.test_labels, self.test_size = self.get_params(self.vocab, self.tag_map, self.dir+'/../data/test/sentences.txt', self.dir+'/../data/test/labels.txt')
            
        if os.path.isfile(self.dir+'/LSTMNUnits.txt') :
            with open(self.dir+'/LSTMNUnits.txt') as f:
                line =  f.read().strip().split('=')
                self.nUnits =  int(line[1])
            
        else:
            self.nUnits = 50
        
        
        if os.path.isfile(self.dir+'/model.pkl.gz') :
            self.model = self.NER()
            self.model.init_from_file(self.dir+'/model.pkl.gz')
        else:
            self.model = self.NER()
            
        self.ner_dict = {"B-geo": "B-Geographical Entity",
                         "B-gpe": "B-Geopolitical Entity",
                         "B-per":"B-Person",
                         "I-geo":"I-Geographical Entity",
                        "B-org":"B-Organization",
                        "I-org":"I-Organization",
                        "B-tim":"B-Time Indicator",
                        "B-art":"B-Artifact",
                        "I-art":"I-Artifact",
                        "I-per":"I-Person",
                        "I-gpe":"I-Geopolitical Entity",
                        "I-tim":"I-Time Indicator",
                        "B-nat":"B-Natural Phenomenon",
                        "B-eve":"B-Event",
                        "I-eve":"I-Event",
                        "I-nat":"I-Natural Phenomenon",
                        "O": "filler word"
                }
            
        
    def get_vocab(self, vocab_path, tags_path):
        vocab = {}
        with open(vocab_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                vocab[l] = i  
        vocab['<PAD>'] = len(vocab) 
        tag_map = {}
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                tag_map[t] = i 
        
        return vocab, tag_map
    
    def get_params(self, vocab, tag_map, sentences_file, labels_file):
        sentences = []
        labels = []
    
        with open(sentences_file) as f:
            for sentence in f.read().splitlines():
                # replace each token by its index if it is in vocab
                # else use index of UNK_WORD
                s = [vocab[token] if token in vocab 
                     else vocab['UNK']
                     for token in sentence.split(' ')]
                sentences.append(s)
    
        with open(labels_file) as f:
            for sentence in f.read().splitlines():
                # replace each label by its index
                l = [tag_map[label] for label in sentence.split(' ')] # I added plus 1 here
                labels.append(l) 
        return sentences, labels, len(sentences)
    
    
    def data_generator(self, batch_size, x, y, pad, shuffle=False):
        num_lines = len(x)
        lines_index = [*range(num_lines)]
        
        if shuffle:
            rnd.shuffle(lines_index)
        
        index = 0
        while True:
            buffer_x, buffer_y  = [], []
            
            max_len = 0
            for i in range(batch_size):
                if index >= num_lines:
                    index = 0
                    if shuffle:
                        rnd.shuffle(lines_index)
                        
                buffer_x.append(x[lines_index[index]])
                buffer_y.append(y[lines_index[index]])
                
                if max_len < len(x[lines_index[index]]):
                    max_len = len(x[lines_index[index]])
                index += 1
            
            X = np.full([batch_size, max_len], pad)
            Y = np.full([batch_size, max_len], pad)
            for i in range(batch_size):
                x_i, y_i = buffer_x[i], buffer_y[i]
                for j in range(len(x_i)):
                    X[i,j] = x_i[j]
                    Y[i,j] = y_i[j]
            #print(X.shape[1])
            yield((X,Y))
            
    def NER(self, vocab_size = None, d_model = None,tag_size = None):
        if vocab_size is None:
            vocab_size = len(self.vocab)
        if tag_size is None:
            tag_size = len(self.tag_map)
        
        if d_model is None:
            d_model = self.nUnits
            
        model = tl.Serial(
              tl.Embedding(vocab_size, d_model), # Embedding layer
              tl.LSTM(d_model), # LSTM layer
              tl.Dense(tag_size), # Dense layer with len(tags) units
              tl.LogSoftmax()  # LogSoftmax layer
          )
          ### END CODE HERE ###
        return model
    
    def trainingModel(self,train_steps = 1):
        
        
        if os.path.isfile(self.dir+'/metrics.txt'):
            os.remove(self.dir+"/metrics.txt")
        if os.path.isfile(self.dir+'/model.pkl.gz'):
            os.remove(self.dir+"/model.pkl.gz")
        
        self.model = self.NER()
        batch_size = 64
        
        train_generator = trax.data.inputs.add_loss_weights(
                self.data_generator(batch_size, self.t_sentences, self.t_labels, self.vocab['<PAD>'],True),
                id_to_mask = self.vocab['<PAD>'])
        
        eval_generator = trax.data.inputs.add_loss_weights(
                self.data_generator(batch_size, self.v_sentences, self.v_labels, self.vocab['<PAD>'],True),
                id_to_mask = self.vocab['<PAD>'])
        
        train_task = training.TrainTask(
                labeled_data = train_generator, 
                loss_layer = tl.CrossEntropyLoss(),
                optimizer = trax.optimizers.Adam(0.01),
                n_steps_per_checkpoint = 10,
                )
        
        eval_task = training.EvalTask(
                labeled_data = eval_generator,
                metrics = [tl.CrossEntropyLoss(), tl.Accuracy()],
                n_eval_batches = 10
                )
        training_loop = training.Loop(
                self.model,
                tasks = train_task,
                eval_tasks = eval_task,
                output_dir = self.dir,
                )
        with open(self.dir+'/metrics.txt', 'w') as f:
            with contextlib.redirect_stdout(f):
                training_loop.run(n_steps = train_steps)
        return len(self.vocab), len(self.t_sentences), len(self.v_sentences)
    
    def predict(self,sentence):
        s = [self.vocab[i] if i in self.vocab else self.vocab['UNK'] for i in sentence.split()]
        batch_data = np.ones((1, len(s)))
        batch_data[0][:] = s
        sentence = np.array(batch_data).astype(int)
        outputs = self.model(sentence)
        outputs = np.argmax(outputs, axis=-1)
        labels = list(self.tag_map.keys())
        pred, predictLabel = [], []
        for i in range(len(outputs[0])):
            idx = outputs[0][i]
            pred.append(labels[idx])
            predictLabel.append(self.ner_dict[labels[idx]])
        return pred, predictLabel
            

# set random seeds to make this notebook easier to replicate
#trax.supervised.trainer_lib.init_random_number_generators(33)



#ner = NERClass()
#ner.trainingModel(train_steps=100)
#sentence = "Peter Navarro, the White House director of trade and manufacturing policy of U.S, said in an interview on Sunday morning that the White House was working to prepare for the possibility of a second wave of the coronavirus in the fall, though he said it wouldn’t necessarily come"
#predictions = ner.predict(sentence)
