Apostolos' Scondrianis Branch for ENSF544 Final Project #2

----------------------------------------------------------
1. Text preprocessing used for our models in text_preprocessing.py
  - Removal of mentions
  - Removal of Emojis
  - Removal of repeated characters in words
  - Removal for duplicate spaces and other types of whitespaces
  - Removal of numbers
  - Removal of URLs
2. Exploratory Data Analysis for ML and Deep Model experiments performed by Travis Dow and Apostolos Scondrianis
3. Working BERT model for classification of Tweet from each country.
  - Model performs with a training accuracy of about 55% on training dataset classifying a tweet between 77 Countries.
  - Model is ran on about 127,000 tweets.
  - Visualizations of the BERT model
  - Confusion Matrices for top 10 countries
  - Model classification report
    - a) Top 10 BERT Model -> 0.5 Dropout  9/20 Epochs/Patience 3
        Loss     : 1.247810959815979
        Accuracy : 0.6123721599578857
    - b) Top 10 BERT Model -> 0.2 Dropout 5/20 Epochs/Patiences 3
        Loss     : Test loss: 1.2485618591308594
        Accuracy : Test accuracy: 0.6129566431045532
    - c) Top 20 BERT Model -> 0.2 Dropout  7/20 Epochs/Patience 3
        Loss     : 1.455275535583496
        Accuracy : 0.586527943611145
    - d) Top 20 BERT Model -> 0.5 Dropout 10/20 Epochs/Patience 3
        Loss     : 1.4809858798980713
        Accuracy : 0.5778112411499023
4. LSTM Model on for classification of Tweets from each country
    - a) Top 10 LSTM Model -> 0.5 Dropout 4/20 Epochs/Patience 3
        Loss     : 1.499
        Accuracy : 0.575
    - b) Top 10 LSTM Model -> 0.2 Dropout 4/20 Epochs/Patience 3
        Loss     : 1.494
        Accuracy : 0.574
    - c) Top 20 LSTM Model -> 0.2 Dropout 4/20 Epochs/Patience 3
        Loss     : 1.766
        Accuracy : 0.531
    - d) Top 20 LSTM Model -> 0.5 Dropout 4/20 Epochs/Patience 3
        Loss     : 1.757
        Accuracy : 0.539
