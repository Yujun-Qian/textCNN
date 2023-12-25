import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
import numpy as np
import random
import math
from collections import Counter
import torch as torch
from transformers import BertTokenizer, BertModel
from transformers import TFAutoModelForSequenceClassification, AutoModelForSequenceClassification, TFBertForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam

device = torch.device("cuda")
pretrained_model = TFBertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

print(tf.config.list_physical_devices('GPU'))

LANGUAGE_TYPE = "br"
CWD = 'model_result2_br_deploy_v2/'
TASK_CLASS_NUMBER = 2  # 2代表二分类， 3 代表多分类
MAX_SEQUENCE_LENGTH = 16
MAX_WORD_LENGTH = 10
POS_SAMPLE_WEIGHT = 1  # 正样本重复倍数
EMBEDDING_CHAR_DIM = 25
LSTM_DIM = 25

query_texts = []
query_labels = []

origin_train_data = CWD + "br_train_binary.txt.full.v2.csv"
if TASK_CLASS_NUMBER == 3:
    origin_train_data = "../data/query/model_data/{}/seg/{}_train_multi_seg.txt".format(LANGUAGE_TYPE, LANGUAGE_TYPE)

def tokenize_dataset(data):
    return tokenizer(data["query"])

dataset = load_dataset("csv", data_files=origin_train_data, split="train")
print(dataset)

tokenized_data = dataset.map(tokenize_dataset)
print(tokenized_data)

train_data = tokenized_data.train_test_split(test_size=0.1)
print(train_data)

tf_train_dataset = pretrained_model.prepare_tf_dataset(
    train_data["train"],
    shuffle=True,
    batch_size=16,
    tokenizer=tokenizer
)

tf_test_dataset = pretrained_model.prepare_tf_dataset(
    train_data["test"],
    shuffle=True,
    batch_size=16,
    tokenizer=tokenizer
)

EMBEDDING_DIM = 100

torch.cuda.empty_cache()

batch_size = 16

val_dataset = load_dataset("csv", data_files="br_val_binary.txt.csv", split="train")
print(val_dataset)

val_data = val_dataset.map(tokenize_dataset)
print(val_data)

tf_val_dataset = pretrained_model.prepare_tf_dataset(
    val_data,
    shuffle=True,
    batch_size=16,
    tokenizer=tokenizer
)

class model_per_epoch(tf.keras.callbacks.Callback):
    def __init__(self, model, filepath, save_best_only):
        self.filepath = filepath
        self.model = model
        self.save_best_only = save_best_only
        self.lowest_loss = np.inf
        self.best_weights = self.model.get_weights()
    def on_epoch_end(self, epoch, logs=None):
        v_loss = logs.get('val_loss')
        print(v_loss)
        if v_loss< self.lowest_loss:
            self.lowest_loss = v_loss
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.model.save_pretrained(self.filepath)
        if self.save_best_only == False:
            name= str(epoch) + '-' + str(v_loss)[:str(v_loss).rfind('.')+3] + '.h5'
            file_id = os.path.join(self.filepath, name)
            self.model.save(file_id)
    #def on_train_end(self, logs=None):
        #if self.save_best_only == True:
            #self.model.set_weights(self.best_weights)
            #name= str(self.best_epoch) +'-' + str(self.lowest_loss)[:str(self.lowest_loss).rfind('.')+3] + '.h5'
            #file_id=os.path.join(self.filepath, name)
            #self.model.save(file_id)
            #print(' model is returned with best weiights from epoch ', self.best_epoch)

def TextCNN_model_2(train, val):
    model = pretrained_model
    model_checkpoint_callback = model_per_epoch(
      model,
      CWD + "model_result2_best/" + LANGUAGE_TYPE + "_textCNN/",
      True
    )

    num_epochs = 1
    num_train_steps = len(tf_train_dataset) * num_epochs
    lr_scheduler = PolynomialDecay(initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps)
    optimizer = Adam(learning_rate=lr_scheduler)
    model.compile(optimizer=optimizer, metrics=["accuracy"])
    model.fit(tf_train_dataset, validation_data=tf_test_dataset, epochs=num_epochs, callbacks=[model_checkpoint_callback])
    model.save_pretrained(CWD + "model_result2/" + LANGUAGE_TYPE + "_textCNN/")
    return model

def TextCNN_model_3(x_train, y_train, x_val, y_val, embedding_matrix):
    return True

if TASK_CLASS_NUMBER == 2:
    load_model = TextCNN_model_2(train_data, val_data)
else:
    TextCNN_model_3(x_train, y_train, x_test, y_test)

if TASK_CLASS_NUMBER == 2:
    load_model = TFBertForSequenceClassification.from_pretrained(CWD + "model_result2_best/" + LANGUAGE_TYPE + "_textCNN/")
    load_model.summary(line_length = 200, positions = [.22, .55, .67, 1.])
else:
    load_model = tf.keras.models.load_model("model_result3/" + LANGUAGE_TYPE + "_textCNN/")

test_dataset = load_dataset("csv", data_files="br_random_query_seg.txt.csv", split="train")
print(test_dataset)

test_data = test_dataset.map(tokenize_dataset)
print(test_data)

tf_test_dataset = load_model.prepare_tf_dataset(
    test_data,
    shuffle=False,
    batch_size=16,
    tokenizer=tokenizer
)

print(test_data[1]['query'])
my_dataset = tf_test_dataset.take(2)
for elem in my_dataset:
    print(elem)


raw = load_model.predict(tf_test_dataset)
print(raw)
# logits used in context of neural networks is different:
# 1. This is the very tensor on which you apply the argmax function to get the predicted class.
# This is the very tensor which you feed into the softmax function to get the probabilities for the predicted classes.
preds = raw["logits"]
result = np.argmax(preds, axis=1)
print(preds.shape, result.shape)

if TASK_CLASS_NUMBER == 2:
    acc_number = 0
    predict_pos_number = 0
    predict_neg_number = 0
    acc_pos_number = 0
    acc_neg_number = 0
    test_pos_number = 0
    test_neg_number = 0

    error_pos_query = []
    error_pos_input = []
    error_pos_score = []
    error_pos_score_1 = []
    for i in range(0, len(result)):
        if test_dataset[i]['label'] == 1:
            test_pos_number += 1
        else:
            test_neg_number += 1

        if result[i] == 1:
            predict_pos_number += 1
            if test_dataset[i]['label'] == 1:
                acc_pos_number += 1
                acc_number += 1
        else:
            predict_neg_number += 1
            if test_dataset[i]['label'] == 0:
                acc_neg_number += 1
                acc_number += 1
            else:
                error_pos_query.append(test_dataset[i]['query'])
                error_pos_score.append(preds[i][1])
                error_pos_score_1.append(preds[i][0])
    for i in range(0, 61):
      print("====" + str(i) + "======")
      print(test_dataset[i]['query'])
      print(preds[i][1])
      print(preds[i][0])

    print("没有召回的正样本sample:")
    for i in range(0, len(error_pos_query)):
        print("query: {},  model score: {}, model_score_other: {}".format(error_pos_query[i], error_pos_score[i], error_pos_score_1[i]))

    print("正样本个数:" + str(test_pos_number) + ", 预测正样本个数" + str(predict_pos_number) + ", 正确预测正样本个数:" + str(acc_pos_number) +
          ", 召回率:" + str(acc_pos_number/test_pos_number) + ", 准确率:" + str(acc_pos_number/predict_pos_number))
    print("负样本个数:" + str(test_neg_number) + ", 预测负样本个数" + str(predict_neg_number) + ", 正确预测负样本个数:" + str(acc_neg_number) +
          ", 召回率:" + str(acc_neg_number/test_neg_number) + ", 准确率:" + str(acc_neg_number/predict_neg_number))
    print("F1 = " + str(2 * acc_pos_number/test_pos_number * acc_pos_number/predict_pos_number / (acc_pos_number/test_pos_number + acc_pos_number/predict_pos_number)))
    print("acc = "  + str(acc_number/(len(result))))
