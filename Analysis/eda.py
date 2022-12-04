import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

data = pd.read_csv("../Data/tweets.csv")

# PIE CHART

# Get country frequency counts
df = data['country'].value_counts()

# Get top 10 countries
df2 = df[:9].copy()

# Others
df3 = df[9:].copy()
df3 = pd.Series(df3.sum(), index=['Other'])

# Combine
df2 = df2.append(df3)

# Make the pie chart
pie_data = df2
cmap = plt.get_cmap("tab10")
inner_colors = cmap(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
explode = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
plt.pie(pie_data, labels=pie_data.index, autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2, startangle=90, rotatelabels=True, colors=inner_colors, explode=explode)
plt.title("Frequency of Country")
plt.show()

'''
# WORD CLOUD

data = data['text']

comment_words = ""
stopwords = set(STOPWORDS)

# Iterate through the csv file
for val in data:
    
    # Typecast each val to string
    val = str(val)

    # Split the value
    tokens = val.split()

    # Convert each token to lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    
    comment_words += " ".join(tokens)+" "

# Make the word cloud
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords, min_font_size=10).generate(comment_words)

# Plot the word cloud
plt.figure(figsize=(8,8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
'''