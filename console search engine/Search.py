import nltk
import os
import sys
import importlib
import time
import porter2stemmer
import string
import re
from math import log, sqrt
from collections import defaultdict  # avoid the hassle of creating a new empty dictionary
from nltk.corpus import stopwords
# import tfidf
# import json
# from os.path import join
# from stemming.porter2 import stem
# from functools import reduce
# from autocorrect import spell
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


non_alphanum = re.compile('[^a-zA-Z0-9_]')
# alphanum = re.compile('[a-zA-Z0-9_]')
# non_whitespace = re.compile('[^ \t\n\r\f\v]')
# whitespace = re.compile('[ \t\n\r\f\v]')
# non_digit = re.compile('[^0-9]')
# digit = re.compile('[0-9]')
stop_words = set(stopwords.words('english'))
index = defaultdict(list)  # return empty list
number_of_documents = 1016
vector_doc = []
vector_docs = {}
equivalent_document_vector = {}
add = ""
importlib.reload(sys)
# sys.setdefaultencoding("UTF-8") or sys.setdefaultencoding("ISO-8859-1")


def address_of_folder():
    global add
    add = input("Please enter your folder address: ")
    while True:
        if add == '1':
            exit()
        elif os.path.isdir(add):
            break
        else:
            add = input("Please enter correct your folder address or press '1' to quit: ")


def search_document():
    global number_of_documents
    a = add
    k = 0
    for file in os.listdir(a):
        k = k + 1
    number_of_documents = k + 1


def loop_for_all_docs():
    a = add
    for file in os.listdir(a):
        text = get_text_document(a, file)
        list2 = get_list(text)
        vector = create_vector(list2)
        vector_doc.append(vector)


def get_list(text):
    global list1, punctuations, table
    punctuations = "?:!.,;@#$%^&*~/`''[]{}|"
    punctuations = string.punctuation.replace("'", "")
    text = text.strip()
    list1 = text.split()
    list1 = text.lower()
    list1 = nltk.word_tokenize(text)
    table = str.maketrans('', '', '\t')
    # ps = nltk.stem.PorterStemmer()
    ps = porter2stemmer.Porter2Stemmer()
    # ps = nltk.stem.WordNetLemmatizer()
    # ps = nltk.stem.LancasterStemmer()
    # ps = nltk.stem.SnowballStemmer('english', ignore_stopwords=True)
    temp = []
    for words in list1:
        if words in punctuations:
            list1.remove(words)
        if words not in punctuations:
            words = words
        if words not in stop_words:
            temp.append(ps.stem(words))
            # temp.append(ps.stem(spell(words))) spell correction for sms text
        if words in stop_words:
            list1.remove(words)
            # temp.append(" ")
    temp = [words.lower() for words in list1 if not words.lower() in stop_words]
    list1 = temp.copy()
    list1 = [words.translate(table) for words in list1]
    list1 = [words for words in list1 if not words.isdigit()]
    list1 = [words for words in list1 if not len(words) == 1]
    list1 = [str for str in list1 if str]
    list1 = [words.lower() for words in list1]
    list1 = [words for words in list1 if len(words) > 2]
    list1 = [non_alphanum.sub('', words) for words in list1]
    list1 = [words for words in list1 if words != '']
    # list1 = [alphanum.sub('', words) for words in list1]
    # list1 = [non_whitespace.sub('', words) for words in list1]
    # list1 = [whitespace.sub('', words) for words in list1]
    # list1 = [non_digit.sub('', words) for words in list1]
    # list1 = [digit.sub('', words) for words in list1]
    return list1
    # return temp
    # return " ".join(list1)


def create_vector(list3):
    vector = {}
    global equivalent_document_vector
    for lis in list3:
        if lis in vector:
            vector[lis] += 1
        else:
            vector[lis] = 1
            if lis in equivalent_document_vector:
                equivalent_document_vector[lis] += 1
            else:
                equivalent_document_vector[lis] = 1
    return vector


def get_text_document(a, file):
    try:
        str1 = open(a+'/'+file).read()
    except:
        str1 = ""
    return str1


def create_vector_from_space(list4):
    vector = {}
    for lis in list4:
        if lis in vector:
            vector[lis] += 1.0
        else:
            vector[lis] = 1.0
    return vector


def create_index():
    for file in os.listdir(add):
        theano = 1
        vect = defaultdict(list)
        f = add+'/'+file
        line = open(f).read().splitlines()
        line = remove_metadata(line)
        # snowball = nltk.stem.PorterStemmer()
        snowball = porter2stemmer.Porter2Stemmer()
        for lin in line:
            lists = nltk.word_tokenize(lin)
            for word in lists:
                vect[snowball.stem(word)].append(theano)
            theano += 1
        vector_docs[file] = vect


def tf_idf():
    for vect in vector_doc:
        length = 0.0
        for word in vect:
            word1 = vect[word]
            temp = calculate(word, word1)
            vect[word] = temp
            length += temp ** 2
        length = sqrt(length)
        if length != 0:
            for word in vect:
                vect[word] /= length


def calculate(word, word1):
    return log(1 + word1) * log(number_of_documents / equivalent_document_vector[word])


def vector_by_tf_idf(space_vector):
    length = 0.0
    for word in space_vector:
        word1 = space_vector[word]
        if word in equivalent_document_vector:
            space_vector[word] = calculate(word, word1)
        else:
            space_vector[word] = log(1 + word1) * log(number_of_documents)
        length += space_vector[word] ** 2
    length = sqrt(length)
    if length != 0:
        for word in space_vector:
            space_vector[word] /= length


def scalar_product(vector_x, vector_y):
    epsilon = 1e-4
    if len(vector_x) >= len(vector_y) - epsilon and len(vector_x) <= len(vector_y) + epsilon:  # len(space_x) < len(space_y)
        temp = vector_x
        vector_x = vector_y
        vector_y = temp
    Sum_x = vector_x.keys()
    Sum_y = vector_y.keys()
    Sum_x_y = 0
    for i in Sum_x:
        if i in Sum_y:
            Sum_x_y += vector_x[i] * vector_y[i]
    return Sum_x_y


def get_data_from_space_vector(space_vector):
    data = []
    a = add
    k = 0
    for file in os.listdir(a):
        top_k = scalar_product(space_vector, vector_doc[k])
        data.append((file, top_k))
        data = sorted(data, key=lambda a: a[1])  # reverse=True
        k = k + 1
    return data


def remove_metadata(line):
    global start
    start = 0
    for i in range(len(line)):
        if line[i] == '\n':
            start = i + 1
            break
    new_line = line[start:]
    return new_line


def main():
    index_time = time.time()
    address_of_folder()
    search_document()
    loop_for_all_docs()
    create_index()
    tf_idf()
    print("Indexing took " + str(time.time() - index_time) + " seconds")
    while True:
        query = input("Enter your query to search: ")
        if len(query) == 0:
            break
        space_list = get_list(query)
        space_vector = create_vector_from_space(space_list)
        vector_by_tf_idf(space_vector)
        answer = get_data_from_space_vector(space_vector)
        for tup in answer:
            print("The document name contain this word is: " + str(tup[0]))
            print("The score is: " + str(tup[1]))
            s = set()
            for word in space_list:
                for number in vector_docs[str(tup[0])][word]:
                    s.add(number)
                for t in s:
                    print("The line number has this word is: ", t)


if __name__ == '__main__':
    space = sys.argv[1:]
    main()