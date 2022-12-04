import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# ---------- Import data and convert categorical data to int ----------

df = pd.read_csv("data.csv")
X = df['text']
countries = df['country']
cat = pd.Categorical(df['country'])
Y = pd.Series(cat.codes)

# ---------- Split data ----------

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=0)

# ---------- Setup BERT ----------

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
print("Setup complete.")

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

preprocess_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocess_text)
print("BERT layers made.")

l = tf.keras.layers.Dropout(0.3, name='dropout')(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
print("NN layers made.")

model = tf.keras.Model(inputs=[text_input], outputs=[l])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=1, batch_size=16)
print("Model made.")

Y_predict = model.predict(X_test)
#Y_predict = Y_predict.flatten()
print("Y_predict:")
print(Y_predict)
print("Y_test:")
print(Y_test)

out = pd.DataFrame(Y_predict).to_csv("predict.csv")
out = pd.DataFrame(Y_test).to_csv("test.csv")