import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

df = pd.read_csv("collated.csv")
print(df.head())