import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub    # repo of trained ML models
import tensorflow_text as text
# Tensors are multi-dimensional arrays with a uniform type (dtype); immutable

df = pd.read_csv("spam.csv", names=["label", "message"])
df.rename(columns = {'label':'Category', 'message':'Message'}, inplace=True)

df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)

X_train, X_test, Y_train, Y_test = train_test_split(df['Message'], df['spam'], stratify=df['spam'], random_state=0)

# Keras is a deep learning API that runs ontop of TensorFlow
# A layer is the basic building block of a neural network in Keras
# A layer consists of a tensor-in tensor-out computation function
# Unlike a normal function, a layer maintains its own state
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3") # Load up pre-saved models
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
print("Setup complete.")

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text') # Used to instantiate a Keras tensor
    # shape = a shape tuple (int) i.e. batches of [int]-dimensional vectors (blank = shape unknown)
    # dtype = expected data type
    # name = name string for the layer
    # Returns a tensor

preprocessed_text = bert_preprocess(text_input) # Pre-saved model used to pre-process plain text into the format expected by BERT
outputs = bert_encoder(preprocessed_text)       # Pre-saved model of a version of BERT implementation
print("BERT layers made.")

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
print("NN layers made.")

# Use inputs and outputs to construct final model
model = tf.keras.Model(inputs=[text_input], outputs=[l])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=2, batch_size=32)
print("Model made.")

Y_predict = model.predict(X_test)
Y_predict = Y_predict.flatten()
print("Y_predict: ")
print(Y_predict)
print("Y_test: ")
print(Y_test)

out = pd.DataFrame(Y_predict).to_csv("predict.csv")
out = pd.DataFrame(Y_test).to_csv("test.csv")