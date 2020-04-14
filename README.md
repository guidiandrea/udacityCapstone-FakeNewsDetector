# udacityCapstone-FakeNewsDetector
A repository made for showing my Udacity Capstone Project, entirely realized on Amazon SageMaker. This was my very first time using SageMaker from end to end.
## Structure
The folder "data" contains data downloaded from Kaggle: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

In the Notebook "Data Exploration and S3 Integration" I am exploring the data and saving the cleaned data to S3.

Then in "Model Building", I tried a Naive Bayes approach which did not go well.

In "Text shaping for RNN and training", I used the tokenized articles to train a Bidirectional LSTM with single sigmoid output, which led me to a test accuracy of ~ 0.985.

In the source_train folder, training scripts for Naive Bayes and LSTM are provided.

The helper.py file contains a couple of useful (and non useful) functions. 

The tokenizer.pkl file contains a pickled version of the fitted Keras tokenizer I used to preprocess data for the LSTM.


