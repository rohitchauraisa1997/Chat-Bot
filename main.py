import json
import pickle
import random
# from pprint import pprint

import nltk
import numpy as np
import tensorflow
import tflearn
# used to stem our words.
from nltk.stem.lancaster import LancasterStemmer

# from utils import chat, bag_of_words
stemmer = LancasterStemmer()

# reading the intents.json file
with open("intents.json", "r") as file:
    data = json.load(file)
    # print(data)

try:
    """
    doing this so that we dont have to do the 
    processing again and again.
    open the previously revised pickled data.
    """
    with open("data.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)
except Exception as error:
    words = list()
    labels = list()
    # docs_x and docs_y for each pattern to
    # get what intent is it a tag of
    docs_x = list()
    docs_y = list()

    for intent in data["intents"]:
        # print(intent)
        for pattern in intent["patterns"]:
            """
            Stemming takes each word thats in our pattern 
            and bring it down to the root word.
            there? --> there
            whats --> what
            """
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            # print("-"*50)
            # print(pattern)
            # print(wrds)
            # print(docs_x)
            # print(docs_y)
            # print(list(zip(docs_x,docs_y)))
            # print("-"*50)

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    """
    stemming all words in words list.
    and removing any duplicates.
    """
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    print("*" * 50)
    words = sorted(list(set(words)))
    print("words", words)
    print("*" * 50)
    print("docs_x", docs_x)
    print("*" * 50)
    print("docs_y", docs_y)
    print("*" * 50)
    labels = sorted(labels)

    """
    creating our training and testing output.
    NN only understand numbers.
    We create a "bag of words", that represents all
    the words in a pattern, and then we use that to 
    train our model.
    [0,0,0,1]
    "a","b","c","d"
    assigns 1 since "d" is there in our pattern.
    """

    training = list()
    output = list()

    # creating the bag of words
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = list()
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            """
            # if the word exists in the current pattern that we r looping through
            """
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        """
        look through the labels list and see where the tag is in that list
        and set that value to 1 in the output_row.
        """
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    with open("data.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)

training = np.array(training)
output = np.array(output)

print("+=" * 50)
print("training:", training)
print("output:", output)
print("+=" * 50)
"""
Building our Model.
"""


tensorflow.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
"""
fitting our model i.e. passing it our training data.
n_epoch is the amount of time its gonna process the same data.
"""

# model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
# model.save("model.tflearn")
try:
    model.load("model.tflearn")
except Exception as error:
    print("error", error)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    """
    helps in making prediction by converting 
    user input into bag of words.
    stemming the words and assigning 1 if word is in 
    "user input" else 0
    """
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


def chat():
    '''
    helps pick the response with highest prob.
    '''
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        # print("results",results)
        results_index = np.argmax(results)
        # print("results_index",results_index)
        tag = labels[results_index]
        # print("tag",tag)
        
        if results[results_index] >0.7:
            for intent_tag in data["intents"]:
                if intent_tag["tag"] == tag:
                    responses = intent_tag["responses"]
            print(random.choice(responses))
        else:
            print("I didnt get that, please try again....")

chat()
