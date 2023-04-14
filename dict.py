import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
import numpy as np
import random
import math
from imblearn.over_sampling import SMOTE
from collections import Counter

word_count = []
query_labels = []
candidates = []

pos_word = []
neg_word = []

with open("pos_processed_word.txt", "r") as file:
    for line in file:
        words = line.split()
        for word in words:
          pos_word.append(word)
#print(np.histogram(candidates, bins=np.arange(20)))

with open("neg_processed_word.txt", "r") as file:
    for line in file:
        words = line.split()
        for word in words:
          neg_word.append(word)

neg_word_dict = dict(enumerate(neg_word))
neg_word_dict = {v: k for k, v in neg_word_dict.items()}
pos_word_dict = dict(enumerate(pos_word))
pos_word_dict = {v: k for k, v in pos_word_dict.items()}

bw_result = open('neg_candidate_word.txt', 'w')
for neg,_ in neg_word_dict.items():
    if len(neg) >= 3 and pos_word_dict.get(neg) == None and neg.startswith("(") == False and neg.endswith(")") == False:
      bw_result.write(neg)
      bw_result.write("\n")
bw_result.flush()
bw_result.close()
