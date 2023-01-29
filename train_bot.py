# In this file we’ll create the model and train it on the dataset prepared.
# 1. Import sequential from ternsorflow.keras.models to create CNN model as we did before. 
# 2. Import Dense, Activation and Dropout layers. Since the model will be trained 
#    on a small dataset, we’ll need less number of layers than we used before. 
# 3. Import Adam optimizer. Optimizers are used to reduce losses while training the model.


#    Create a function for defining the CNN model. This function train_bot_model() takes 
#    train_x and train_y as parameters.
# 1. Define the model being sequential using the Sequential() method. 
# 2. The very first layer we’ll add is a Dense layer with 128 output units. 
#    Input will be the training data that is train_x and activation function as ‘relu’. 
# 3. Add Dropout layer with 0.5 dropout. 4. The second layer is also a Dense layer with 
#    64 output units. Activation function as ‘relu’. 
# 5. Add Dropout layer with 0.5 dropout. 
# 6. The last layer will be the dense layer with output units equal to the number of tags.
# Use softmax activation function as its last layer of our model. Compile, fit and save the 
# model. we'll follow these steps in the similar manner as we have done before. 
# 7. Compile the model by defining loss, optimizer and metrics. 
# 8. The next step would be to fit and save the model. Provide all the necessary parameters 
# to fit the model. (training data, no of epochs, batch size and verbose) 
# 9. Save the model by name chatbot_model.h5. 
# 10.After saving the model file print a message ‘Model File Created & Saved’.




# Model Training Lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

from data_preprocessing import preprocess_train_data

def train_bot_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile Model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])

    # Fit & Save Model
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=True)
    model.save('chatbot_model.h5', history)
    print("Model File Created & Saved")


# Calling Methods to Train Model
train_x, train_y = preprocess_train_data()

train_bot_model(train_x, train_y)

