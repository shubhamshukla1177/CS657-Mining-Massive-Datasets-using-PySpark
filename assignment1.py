#!/usr/bin/env python
from pyspark.sql import SparkSession, Row
from pyspark import SQLContext, SparkFiles, SparkConf, SparkContext
import requests
from pyspark.sql.functions import split,explode,col,regexp_replace,udf,explode,substring,year,avg,stddev,when,stddev_pop,trim,map_keys, map_values
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, IntegerType, ArrayType, MapType
import re
import numpy
from pyspark.sql.functions import collect_list, size
from itertools import combinations
import sys

spark = SparkSession.builder.appName('Assignment1').getOrCreate()
#Create a spark context in session
sc = spark.sparkContext
#Reduce output by only showing me the errors
sc.setLogLevel("ERROR")

def read_data(file_path):
  return spark.sparkContext.textFile(file_path)
  
file_path =  '/user/kfnu/input/stateoftheunion1790-2021.txt'
data = read_data(file_path)
rdd_with_index = data.zipWithIndex()
filtered_rdd = rdd_with_index.filter(lambda x: x[1] >= 247)
final_rdd = filtered_rdd.map(lambda x: x[0])    
  
def normalizeWords(text):
    # Remove HTML commands
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URL addresses (http/https/ftp/www)
    text = re.sub(r'http\S+|www\S+|ftp\S+', '', text)
    # Remove non-alphanumeric characters (except for single quotes)
    text = re.sub(r"[^a-zA-Z0-9']", ' ', text)
    text = re.sub(r"'", '', text)
    text = re.sub(r'[.,!?;:]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Split the text into words based on whitespace
    #words = text.split()
    # Define a list of stopwords to remove (e.g., 'to', 'for', etc.)
    #stopwords = set(['to', 'for', 'and', 'the', 'in', 'on', 'a', 'an', 'at', 'of','it', 'this', 'that', 'he', 'she', 'i', 'their', 'our', 'or', 'but', 'my','by','your', 'they'])  # Add more stopwords as needed
    stopwords = set(['out', 'we', 'was', 'how', 'myself', 'for', 'they', 'about', "hasn't", 'then', 'both', 'so', 're', 'don', 'm', 'as', 'any', 'mightn', 'after', 'you', 'wouldn', 'why', 'been', 'where', 'by', "isn't", 'yourself', 'wasn', 'a', "haven't", 'did', "hadn't", 'their', 'hasn', 'doing', 'be', 'further', 'ours', 'now', 'am', 'her', "you'll", 'yourselves', 'that', 'my', 'what', 'to', 'd', 'not', "won't", "couldn't", 'own', 'there', 'this', 'each', 'all', 'haven', 'more', 'me', 've', 'weren', 'which', 'himself', 'nor', 'other', "shouldn't", 'who', "should've", 'same', 'at', 'such', 't', 'up', 'than', 'can', "you've", 'too', 'these', 'while', "wasn't", 'ourselves', 'before', 'i', 'he', "didn't", 'our', 'its', 'but', 'with', "wouldn't", 'those', 'because', 'the', 'y', 'shouldn', 'it', 'mustn', 'hers', 'just', 'doesn', 'ain', 'between', 'over', 'had', 'aren', "mightn't", 'does', 'have', 'and', 'or', 'some', "mustn't", 'only', 'won', 'when', 'needn', 'below', 'in', 'if', 'theirs', "needn't", "aren't", 'isn', 'again', 'his', 'whom', 'll', 'hadn', 'above', 'should', 'itself', 'themselves', 'until', 'are', 'she', 'no', 'from', 'into', 'will', 'your', 'few', 'herself', 'of', 'has', 'down', 'were', 'once', 'ma', 'having', 'them', 'under', 'him', 'shan', 'couldn', 'do', 'on', 'an', "you'd", 'yours', 'being', 'off', 'o', "that'll", 'very', "weren't", 'didn', 'through', "you're", 'most', 'against', "it's", "doesn't", 'here', 'is', 's', "don't", "shan't", 'during', "she's"])
    # Filter out stopwords
    words = [word for word in text if word not in stopwords]
    return words

sentence = final_rdd.flatMap(normalizeWords)
filtered_sentences_rdd = sentence.map(lambda x: x.split('.'))
def generate_pairs(words):
    pairs = []
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            pairs.append((tuple(sorted([words[i], words[j]])), 1))
    return pairs

def occ_pairs(words):
    pairs = []
    words = list(words)
    for i in range(len(words)-1):
        pairs.append((tuple(sorted([words[i], words[i+1]])), 1))
    return pairs

num_sentences = filtered_sentences_rdd.count()
print("total sentences", num_sentences)

word_freqs = filtered_sentences_rdd.flatMap(lambda sentence: set(' '.join(sentence).split())) \
                          .map(lambda word: (word, 1)) \
                          .reduceByKey(lambda a, b: a + b) \
                          .mapValues(lambda count: count / num_sentences)

word_freqs_dict = word_freqs.collectAsMap()
broadcasted_word_freqs = sc.broadcast(word_freqs_dict)

word_pairs_rdd = filtered_sentences_rdd.flatMap(lambda sentence: occ_pairs(set(' '.join(sentence.split()))))

word_pair_counts = word_pairs_rdd.reduceByKey(lambda a, b: a + b) \
                                 .mapValues(lambda count: count / num_sentences)

new_word_pair_counts = word_pair_counts.filter(lambda x: x[1] > 10 / num_sentences)

def compute_lift(pair_freq):
    (word_a, word_b), pab = pair_freq
    pa = broadcasted_word_freqs.value[word_a]
    pb = broadcasted_word_freqs.value[word_b]
    return (pair_freq[0], pab / (pa * pb))

lifts = new_word_pair_counts.map(compute_lift)

high_lift_pairs = lifts.filter(lambda pair_lift: pair_lift[1] > 3)
high_lift_pairs.saveAsTextFile('/user/kfnu/output/high_lift_pairs.txt')