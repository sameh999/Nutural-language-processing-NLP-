<!-- @format -->

# Nutural-language-processing Project 2 

## Goal:

This assignment aims to implement classification strategy for sample (five) different book and
sample (five) different authors and Separate and set aside unbiased random partitions for
training, validation and testing to produce classification predictions and compare them;
analyze the pros and cons of algorithms and generate and communicate the insights and to
know the style for each other so when give the model any document from other book can
detect the other.

## Dataset:

We make search in Gutenberg digital books to collect this book to be and the same category and for
different author to make a difficult to the model to predict the correct labeled data.

1. Data preprocessing:
1. Clean data:

We get the books by URL and make preparation and cleaning processes by download the
stop word from NLTK (remove stop words, punctuation and numbers) and get word only to
help us making classification and change all upper case word to lower case.
**Label data :**
We make labeling for each partition in the book regarding author to could make prediction
according to this lapel.
**Partition data:**
After cleaning processes, we start prepare data by splitting each book to 200 document and
each document have at least 100 word. 2. Text transformation
**BOW (1 gram):**
Make a tokenization for book could make bag of word (BOW) and can enter the
document to the model to know the author of partition of written.
**BoW + 2gram:**
We apply the bag of word with 2 gram we get this matrix of data and we will enter it to
model and will know the accuracy of prediction.
**TF-IDF:**
Using tf-idf of the of document and give to the model to predict the document belong to any
autho
We make a stemming to all word in the 5 books to decrease the word we work on it and we
will do the opration to the trainning data and test date that will enter the model to predict. 3. Classification:
We will divide the data random to training data and test data

## 2- KNN:

applying knn in all data come from

**TF-IDF (1 gram):**

**TF-IDF (2 gram):**

**BOW Unigram:**

**BOW 2 gram :**

## 3- SVM:
pplying SVM in all data come from
**TF-IDF (1 gram):**

**TF-IDF (2 gram):**

**BOW Unigram:**

**BOW 2 gram :**
## 4. Evaluation.
## 5. Error Analysis
   The introduction paragraphs at the beginning and the ending of each
   book are one issue we encountered, and it was already mentioned in the
   data cleansing part. For all works, they are standard paragraphs given
   by Project Gutenberg. Those paragraphs had a negative impact on our
   model, and by deleting them, we were able to improve its accuracy
## 6. Models Insights and The Champion One:

## 7. Bias and Variance
   The Bias: The bias is a measure of how close the model can capture
   the mapping function between inputs and outputs
   The Variance: The variance of the model is the amount the
   performance of the model changes when it is fit on different training
   data.
   The bias and the variance of a model’s performance are connected
   and we calculate them in our project:
