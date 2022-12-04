# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import random
import nltk
import json
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy

lemmatizer = WordNetLemmatizer()
ignore_letters = ["!","?",",","."]


def load_training_text():
    #Load json file containing the patterns, tags, respondses and context_set
    intents = json.loads(open("intents.json").read())
    words = []
    classes = []
    documents = []
    # Load information from json file
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            word_list = [word.lower() for word in word_list]
            words.extend(word_list)
            documents.append((word_list,intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])           
    #print("words: ",words,"\n\n\n documents: ",documents,"\n\n\n classes: ",classes,"\n\n\n")
    
    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    #print(words)
    return words, classes, documents

def load_answer(clas):
    intents = json.loads(open("intents.json").read())
    for intent in intents["intents"]:
        if clas==intent["tag"]:
            return random.choice(intent["responses"])
    

def create_training_data(words, classes, documents):
    training_x = np.zeros((len(documents),len(words)))
    training_y = np.zeros((len(documents),len(classes)))
    
    i=-1
    for words_in_doc, clas in documents:
        #print(words_in_doc,clas)
        i+=1
        training_y[i,classes.index(clas)] = 1
        for word in words_in_doc:
            if word not in ignore_letters:
                word = lemmatizer.lemmatize(word)
                training_x[i,words.index(word)] += 1
    return training_x, training_y

def model(num_clases,train_x,train_y,train="True"):
    model = Sequential()
    model.add(Dense(128, input_shape=(train_x.shape[1],),activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_clases,activation="softmax"))
    
    sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=sgd, metrics=["accuracy"])
    if train == "True":
        model.fit(train_x,train_y,epochs=200,shuffle="True",verbose=1)
        model.save("chatbot_model.model")
        return model
    model.load("chatbot_model.model")
    return model

def predict_text(model,text,words,classes):
    test_x = np.zeros((1,len(words)))
    words_in_sentence = nltk.word_tokenize(text)
    for word in words_in_sentence:
        word = lemmatizer.lemmatize(word.lower())
        if word not in words or word in ignore_letters:
            print("words ommited in test: ",word)
            continue
        test_x[0,words.index(word)] = 1
    y = model.predict(test_x)
    max_value = np.max(y,axis=1)
    indices = []
    for value, res in zip(max_value,y):
        comp = value==res
        indices.append(list(comp).index(True))
    return indices
    

if __name__ == "__main__":
    print("hello world")
    
    words, classes, documents = load_training_text()
    nb_classes = len(classes)
    train_x, train_y = create_training_data(words, classes, documents)
    
    model = model(nb_classes,train_x, train_y,train="True")
    
    #test_text = ["at what time are you open?","Hi, how are you","opening hours"]
    
    text = str(input("Start writing to the chatbot. (press q to stop)\n"))
    while text!="q":
        indexes = predict_text(model,text, words, classes)        
        answer = load_answer(classes[indexes[0]])
        text = str(input("{} (press q to stop)\n".format(answer)))
    print("\n good bye world")
    
