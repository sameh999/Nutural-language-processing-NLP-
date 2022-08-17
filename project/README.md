<!-- @format -->

# Nutural-language-processing

# E-commerce Intelligent bot

# Problem Formulation:

There is a need for a chatbot app for smartphones that allows users to have online conversation. Based on the information provided by the user, the chatbot can recommend which medications to take. The development of the application has the potential to improve consumer-pharmacy communication while also assisting the pharmacy in developing a better customer management system.

# Goal:

A chatbot system that not only can understand and analyze the clients’ symptoms but also diagnose the disease as well as recommend the suitable drugs based on machine learning algorithms and Natural Language Processing techniques.

# Project Pipeline

1. Data Collection
   We use EPAR for medicines open-source dataset and (EPAR) European public assessment medical reports from European Medicines Agency (EMA) to build our system.
2. Dataset Features
   Description of Some features:
   • Category: Human or Veterinary
   • Medicine name: about 600 unique drugs
   • Therapeutic area: disease like Anaemia, Diabetes, Bacterial Infections and more
   • International non-proprietary name (INN) / common name
   • Patient safety: a binary feature with yes or no
   • Orphan medicine: a binary feature with yes or no
   • Condition / indication: text descriptions of drugs and diseases
   • URL: for each drug
   We will use “Condition / indication” feature for NLP stages and “Medicine name”, “Therapeutic area” and other features for fulfilment.
3. Exploratory Data analysis
   We apply a word cloud EDA on the “Condition / indication” feature and we got most frequent words as shown in the figure.
   This bar chart shows the frequencies of each disease with repect to number of drugs. We will use the most 20 diseases and their drugs in future steps.
   After applying some filter and explore most feature manually, we found that this data set has many blank or missing values and some in consistent values. We will deal with these issues in the data preparation step.
4. Text Transformation and Feature Engineering
   We implement 2 approaches of text transformation:

• TF-IDF:
Bag of word doesn't capture the importance of the word it gives you the frequency of the word. TF-IDF resolves this matter through computation of two values. TF is count of occurrences of the word in a document. IDF of the word across a set of documents. It tells us how common or rare a word is in the entire document set. The closer it is to 0, the more common is the word. This metric can be calculated by taking the total number of documents, dividing it by the number of documents that contain a word, and calculating the logarithm. We then multiply these two values TF and IDF. We used sklearn tf-idf-transformer which transforms a count matrix to a normalized tf or tf-idf representation.

• LDA:
Latent Dirichlet Allocation (LDA) algorithm is an unsupervised learning algorithm that attempts to describe a set of observations as a mixture of distinct categories. LDA is most commonly used to discover a user-specified number of topics shared by documents within a text corpus. Here each observation is a document, the features are the presence (or occurrence count) of each word, and the categories are the topics. Since the method is unsupervised, the topics are not specified up front, and are not guaranteed to align with how a human may naturally categorize documents. The topics are learned as a probability distribution over the words that occur in each document. Each document, in turn, is described as a mixture of topics.

5. Classification Algorithms:
   SVM
   KNN
   DT
6. Clustering Algorithms
   K-Means + LDA
   We implement 3 algorithms:
   • K-Means:
   The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:

   The centroids of the K clusters, which can be used to label new data
   Labels for the training data (each data point is assigned to a single cluster)

## Error Analysis

 Remove the very beginning and the very last paragraphs of each book
 Remove the most frequent and the least weighted words from TF-IDF Transformation
 Try different collections of books

## Deployment
