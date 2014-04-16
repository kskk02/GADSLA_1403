
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from math import sqrt
from itertools import izip


def cos(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))


def process_text(text):
    text=text.lower()
    return ("".join(c for c in text if c==" " or c.isalpha())).split()
        
# BEGIN TESTS
assert process_text('Python is SO AWESOME!!!!!! YAy!!!!@ I love programming in python!') == ['python', 'is', 'so', 'awesome', 'yay', 'i', 'love', 'programming', 'in', 'python']
# END TESTS 


# Question 2: Count word occurences
def count_words(text):
#    '''FILL IN ANSWER HERE.'''
    listofwords = process_text(text)
    result = dict()
    for word in listofwords:
        if word in result:
            result[word]+=1
        else:
            result[word]=1
    return result # can this be done by list comprehension?


#   return Counter(listofwords)
# BEGIN TESTS
assert count_words('Python is SO AWESOME!!!!!! YAy!!!!@ I love programming in python!') == {'yay': 1, 'python': 2, 'is': 1, 'programming': 1, 'i': 1, 'so': 1, 'in': 1, 'love': 1, 'awesome': 1}
# END TESTS

# Question 3: Create a string distance function
def distance(text1, text2):

    vectorizer = TfidfVectorizer(min_df=1)
    vectorized_text = vectorizer.fit_transform([text1 , text2])
    return (1 - cos(vectorized_text.toarray()[0], vectorized_text.toarray()[1]))

# BEGIN TESTS
assert distance('I love my mom', 'i love my daddy') < distance('I love my mom', 'I am a big boy now')
assert distance('some strings are similar to other strings', 'some strings') > distance('some strings are similar to other strings', 'some strings are similar')
assert distance('i hate hate hate noodles.', 'i hate hate noodles') < distance('i hate hate hate noodles.', 'i hate noodles.')
# END TESTS

print 'All tests passed. Congratulations!'

