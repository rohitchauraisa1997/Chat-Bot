from pprint import pprint
import json
import nltk
# used to stem our words.
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import random
import tflearn

stemmer = LancasterStemmer()

# reading the intents.json file
with open("intents.json", "r") as file:
    data = json.load(file)
    # print(data)

words = list()
labels = list()
#docs_x and docs_y for each pattern to
#get what intent is it a tag of
docs_x = list()
docs_y = list()

for intent in data["intents"]:
    # print(intent)
    for pattern in intent["patterns"]:
        '''
        Stemming takes each word thats in our pattern 
        and bring it down to the root word.
        there? --> there
        whats --> what
        '''
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
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

'''
stemming all words in words list.
and removing any duplicates.
'''
words = [stemmer.stem(w.lower()) for w in words]
print("*"*50)
words = sorted(list(set(words)))
print("words",words)
print("*"*50)
print("docs_x",docs_x)
print("*"*50)
print("docs_y",docs_y)
print("*"*50)
labels = sorted(labels)

'''
creating our training and testing output.
NN only understand numbers.
We create a "bag of words", that represents all
the words in a pattern, and then we use that to 
train our model.
[0,0,0,1]
"a","b","c","d"
assigns 1 since "d" is there in our pattern.
'''

training = list()
output = list()

# creating the bag of words
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = list()
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        '''
        # if the word exists in the current pattern that we r looping through
        '''
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    '''
    look through the labels list and see where the tag is in that list
    and set that value to 1 in the output_row.
    '''    
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1  
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)
