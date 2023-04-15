# tfidf-Algorithm
This article was published as a part of the Data Science Blogathon.

### Overview
In NLP, *tf-idf* is an important measure and is used by algorithms like cosine similarity to find documents that are similar to a given search query.

We write a simple Python program that uses TfidfVectorizer to calculate tf-idf and manually validate this. 

`
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
skl_output = vectorizer.transform(corpus)
`

What is Term Frequency (tf)
tf is the number of times a term appears in a particular document. 

** tf(t) = No. of times term ‘t’ occurs in a document 

tf(t) = (No. of times term ‘t’ occurs in a document) / (No. Of terms in a document)
** 
OR

** tf(t) = (No. of times term ‘t’ occurs in a document) / (Frequency of most common term in a document) **
 
Inverse Document Frequency (idf)
idf is a measure of how common or rare a term is across the entire corpus of documents. So the point to note is that it’s common to all the documents. If the word is common and appears in many documents, the idf value (normalized) will approach 0 or else approach 1 if it’s rare. A few of the ways we can calculate idf value for a term is given below

idf (t) = 1 + log e [ n / df(t) ]

OR

idf(t) = log e [ n / df(t) ]

where

n = Total number of documents available

t = term for which idf value has to be calculated

df(t) = Number of documents in which the term t appears

** idf(t) = log e [ (1+n) / ( 1 + df(t) ) ] + 1 (default i:e smooth_idf = True)

idf(t) = log e [ n / df(t) ] + 1 (when smooth_idf = False)
**

Refer - [Sklearn Official Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
