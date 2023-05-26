import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('C:/Users/medob/Desktop/chatbot/chatbot/Include/intents.json').read())

words = []#hn7ot feha el tokenized patterns 
classes = []#hn7ot feha eltags zy el greetings wkda
documents = []#hn7ot feha tuple ma2soom goz2een goz2 el klam el m3molo tokenize w el goz2 eltany hykon el tag bt3hom
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)#takes content to the list not list to list
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
#set 3shan mykonsh fe dups
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))#3shan n3ml save fe file
pickle.dump(classes, open('classes.pkl', 'wb'))#3shan n3ml save fe file

training = []#hn7ot feha o aw 1 3la 7sb el klma de leha 3laka bl pattern el e7na feeh wla la
outputEmpty = [0] * len(classes)

#el for loop de 3shan n7ot kol el data el lsa han3mlha training fe el list bta3t el training
for document in documents:
    bag = []
    wordPatterns = document[0]#0 3shan bna5od el pattern ama el tag fe index 1
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

#training el data el lsa 3amlnha fe el for loop el foo2
random.shuffle(training)
training = np.array(training)

# x and y values to train the nural network
trainX = training[:, :len(words)]#l8ait 0
trainY = training[:, len(words):]#l8ait 1

#start building the neural network
model = tf.keras.Sequential()#by3ml linear stack of layers each one connected to the next one
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
#dense layer each nuron connected to every nurone even the one before it and 128 da no of layers 
#w el input shape by3ml define el shap bta3 el input data lel first layer bl lenght bta3 el train x
#relu used in hidden layers of the NN better because efficient
model.add(tf.keras.layers.Dropout(0.5))#used to reduce over fitting by making 50% of data values set to 0
model.add(tf.keras.layers.Dense(64, activation = 'relu'))#added another dense layer of 64 layer
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))#softmax btsht8l 3la eloutput enha bt3ml sum lel 
#probability bta3t el classes bta3tna 3shan n3rf el output htkoon ehh

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)#momentum and nestrov bysht8lo 3la ta7seen ada2 el training w eno ykon asra3
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])#loss function

#now trining starts
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')