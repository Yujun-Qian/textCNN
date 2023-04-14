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
pos = []
neg = []

with open("pos_candidate.txt", "r") as file:
    for line in file:
      pos.append(line.strip())
print(len(pos))

with open("neg_candidate_word.txt", "r") as file:
    for line in file:
      neg.append(line.strip())

neg_dict = dict(enumerate(neg))
print(neg_dict.get(0))
print(neg_dict.get(len(neg) - 1))
print(neg_dict.get(len(neg)))
print(random.sample(range(0, 2), 1))
print(random.sample(range(0, 2), 1))
print(random.sample(range(0, 2), 1))
print(random.sample(range(0, 2), 1))

#print(np.histogram(candidates, bins=np.arange(20)))

bw_result = open('cat_result.txt', 'w')
for word in pos:
    count_rand = random.sample(range(0, 2), 1)
    if count_rand[0] == 0:
      count = 6
    else:
      count = 7
    pos_rand = random.sample(range(0, len(neg)), count)
    for position in pos_rand:
      neg_word = neg_dict.get(position)
      prepost_rand = random.sample(range(0, 2), 1)
      if prepost_rand[0] == 0:
        cat_word = neg_word + " " + word
      else:
        cat_word = word + " " + neg_word
      bw_result.write(cat_word + "\n")
bw_result.flush()
bw_result.close()
