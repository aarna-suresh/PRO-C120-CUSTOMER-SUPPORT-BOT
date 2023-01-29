Download all the files in a folder and run the file in the following order. 
1. Create a virtual environment and install all the necessary libraries such as 
nltk, tensorflow and numpy. 
2. Now run the data_preprocessing file for creating the 
training dataset.(train_x and train_y). 
3. Run train_bot file for creating and saving the model file. Next, create a file called predict_response to take user input 
and give a response through the chatbot.

1. Import nltk. From nltk download punkt and wordnet.
2. Download json, numpy, random and pickle. These files are necessary for processing data. 
3. Create a list of ignore_words. 4. Import tensorflow. 5. Now the user input should also be converted into an array of stem words. Import get_stem_words from data_preprocessing. 6. We will load the model file. 7. Load all the data files. The intents.json has the raw dataset. 8. The classes.pkl and words.pkl files have the preprocessed training dataset.
Now write a function to preprocess the text input we get from the user: 1. Create a function called preprocess_user_input. This function takes user input, tokenizes it, converts the tokenized text into stem words. 2. The sorted list of stem words is stored in input_word_token_2. 3. Now, the Bag Of words is created using this list. If the word in the given sentence is present in the list of stem words then ‘1’ is appended in bag_of_words otherwise ‘0’ is appended. 4. This bag_of_words is then converted into a numpy array. Create another function for the prediction of class label. This label will be the predicted tag. So, the response will be chosen from the predicted tag: 1. Create a bot_class_prediction function. This function takes user input as its parameters. 2. Inside this function, the preprocess_user_input function is being called that gives the preprocessed text for user_input. 3. Use the predict function to predict the label and store it in the prediction variable. 4. This prediction variable has an
array of predicted classes with their probabilities. The maximum value is found by using the argmax() function. 5. So, this function returns the prediction as to the class label or tag. At last, create a function to give a response according to the user input. 1. In this function call the function for prediction to get the predicted_class_label. 2. predicted _class is the predicted tag. Now to get the predicted tag we’ll use the classes.pkl file in which we had stored these tags with respective labels. 3. Next, loop through the intents in the intents.json file to compare the predicted tag with all the tags. 4. bot_response is chosen randomly amongst the available responses under the predicted tag. 5. Thus, whenever the user gives an input, it is being checked under which tag it falls and accordingly the response is given from bot.
5. response variable stores the chatbot response. Print this response. 6. Since we have used a while loop, the process will continue. Let’s save and run this file to check how the chatbot works




#Text Data Preprocessing Lib
import nltk

import json
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):

    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words) 
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    bag=[]
    bag_of_words = []
   
    # Input data encoding 
    for word in words:            
        if word in input_word_token_2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)
  
    return np.array(bag)

def bot_class_prediction(user_input):

    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label


def get_bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
   predicted_class = classes[predicted_class_label]

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
        bot_response = random.choice(intent['responses'])
        return bot_response

print("Hi I am Stella, How Can I help you?")

while True:
    user_input = input("Type your message here:")
    print("User Input: ", user_input)

    response = get_bot_response(user_input)
    print("Bot Response: ", response)

