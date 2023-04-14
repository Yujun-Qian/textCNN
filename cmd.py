import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import random
import math
import subprocess
from imblearn.over_sampling import SMOTE
from collections import Counter


words = []

with open("word.txt", "r") as file:
    for line in file:
        words.append(line.strip())

#print(np.histogram(candidates, bins=np.arange(20)))


for word in words:
    cmd_str = "echo \"" + word + "\" | ./i18n_segment br 0 1"
    subprocess.run(cmd_str, shell=True)
    #bw_result.write(p.stdout.readlines())
