#!/usr/bin/env python
# coding: utf-8

# In[26]:


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
print(tf.config.list_physical_devices('GPU'))


# In[27]:


LANGUAGE_TYPE = "in"
TASK_CLASS_NUMBER = 2  # 2代表二分类， 3 代表多分类
MAX_SEQUENCE_LENGTH = 16
POS_SAMPLE_WEIGHT = 1  # 正样本重复倍数


# In[28]:



query_texts = []
query_labels = []

origin_train_data = "../data/query/model_data/{}/{}_train_binary.txt".format(LANGUAGE_TYPE, LANGUAGE_TYPE)
if TASK_CLASS_NUMBER == 3:
    origin_train_data = "../data/query/model_data/{}/seg/{}_train_multi_seg.txt".format(LANGUAGE_TYPE, LANGUAGE_TYPE)
with open(origin_train_data, "r") as file:
    for line in file:
        tokens = line.strip().split("\t")
        if len(tokens) != 2:
            filter_number += 1
            continue
        label = int(tokens[1].strip())
        query = tokens[0].strip()

        query_texts.append(query)
        query_labels.append(label)

print("query texts len {}, query labels len {}".format(len(query_texts), len(query_labels)))




# In[29]:


tokenizer=tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(query_texts)
word_index=tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[30]:


# 打印提取的样本
f = open(LANGUAGE_TYPE + '_' + str(TASK_CLASS_NUMBER) + '_use_data.txt', 'w')
for i in range(0, len(query_texts)):
    f.write(query_texts[i] + '\t' + str(query_labels[i]) + '\n')
f.flush()
f.close()
print("OK")


# In[31]:


#加载embedding 向量

EMBEDDING_DIM = 100
embeddings_index = {}

has_word_index = len(word_index)

embedding_files = '../data/embedding/' + LANGUAGE_TYPE + '_word2v.vec'
f = open(embedding_files)
for line in f:
    values = line.split()
    if len(values) != 101:
        continue
    word = values[0].strip()
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    if word not in word_index:
        has_word_index += 1
        word_index[word] = has_word_index
f.close()

print('Found %s word vectors.' % len(embeddings_index))
print("extend word index number is:%s." % len(word_index))
print('has_word_index:%s.' % has_word_index)

bw_result = open('i18_q_sex_classify_' + LANGUAGE_TYPE + '_' + str(TASK_CLASS_NUMBER) + '_word_index_dict.txt', 'w')
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    bw_result.write(word + "\t" + str(i) + '\n')
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
bw_result.flush()
bw_result.close()
print(embedding_matrix[15])


# In[32]:


x_train_ori, x_test_ori, y_train_ori, y_test_ori = train_test_split(query_texts, query_labels, test_size=0.1)


if POS_SAMPLE_WEIGHT > 1:
    # 对train 样本中的正样本扩大n倍
    x_train_ori_extend = []
    y_train_ori_extend = []
    for i in range(len(y_train_ori)):
        if y_train_ori[i] > 0:
            for j in range(POS_SAMPLE_WEIGHT):
                x_train_ori_extend.append(x_train_ori[i])
                y_train_ori_extend.append(y_train_ori[i])
        else:
            x_train_ori_extend.append(x_train_ori[i])
            y_train_ori_extend.append(y_train_ori[i])

    sample_extend = []
    for i in range(len(x_train_ori_extend)):
        pair_tmp = (x_train_ori_extend[i], y_train_ori_extend[i])
        sample_extend.append(pair_tmp)
    random.shuffle(sample_extend)
    x_tmp = []
    y_tmp = []
    for (k, v) in sample_extend:
        x_tmp.append(k)
        y_tmp.append(v)
    x_train_ori = x_tmp
    y_train_ori = y_tmp


before_test_number = len(x_test_ori)
print(type(x_test_ori))

x_train_ori_set_tmp = set(x_train_ori)
print(len(x_train_ori_set_tmp))

for i in range(0, len(x_test_ori)):
    if i >= len(x_test_ori):
        continue
    if x_test_ori[i] in x_train_ori_set_tmp:
        x_test_ori.pop(i)
        y_test_ori.pop(i)
        i -= 1
after_test_number = len(x_test_ori)

print("test number before {}, after {}".format(before_test_number, after_test_number))


if TASK_CLASS_NUMBER == 2:
    train_pos_number = 0
    train_neg_number = 0
    test_pos_number = 0
    test_neg_number = 0
    for i in y_train_ori:
        if i == 0:
            train_neg_number += 1
        elif i == 1:
            train_pos_number += 1
    for i in y_test_ori:
        if i == 0:
            test_neg_number += 1
        elif i == 1:
            test_pos_number += 1
    print("train number is %s, contains pos number is %s, neg number is %s." % (len(y_train_ori), train_pos_number, train_neg_number))
    print("test number is %s, contains pos number is %s, neg number is %s." % (len(y_test_ori), test_pos_number, test_neg_number))
else:
    train_pos_2_number = 0
    train_pos_1_number = 0
    train_neg_number = 0
    test_pos_2_number = 0
    test_pos_1_number = 0
    test_neg_number = 0
    for i in y_train_ori:
        if i == 0:
            train_neg_number += 1
        elif i == 1:
            train_pos_1_number += 1
        elif i == 2:
            train_pos_2_number += 1
    for i in y_test_ori:
        if i == 0:
            test_neg_number += 1
        elif i == 1:
            test_pos_1_number += 1
        elif i == 2:
            test_pos_2_number += 1
    print("train number is %s, contains pos 2 number is %s, pos 1 number is %s, neg number is %s." % (len(y_train_ori), train_pos_2_number, train_pos_1_number, train_neg_number))
    print("test number is %s, contains pos number is %s, pos 1 number is %s, neg number is %s." % (len(y_test_ori), test_pos_2_number, test_pos_1_number, test_neg_number))

print(x_train_ori[0])
print(x_test_ori[0])


# In[33]:




x_train_sequences = tokenizer.texts_to_sequences(x_train_ori)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')

x_test_sequences = tokenizer.texts_to_sequences(x_test_ori)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')

if TASK_CLASS_NUMBER == 2:
    y_train = np.asarray(y_train_ori)
    y_test = np.asarray(y_test_ori)
else:
    y_train = tf.keras.utils.to_categorical(y_train_ori)
    y_test = tf.keras.utils.to_categorical(y_test_ori)

print(x_train[0])
print(y_train[0])
print(x_test[0])
print(y_test[0])


# In[9]:


# x_train y_train
print(Counter(y_train))

x_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
print(Counter(y_resampled))

x_train = x_resampled
y_train = y_resampled





# In[34]:


print(x_train[-10:-1])
print(y_train[-10:-1])


# In[35]:


def TextCNN_model_2(x_train, y_train, x_val, y_val, embedding_matrix):
    main_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input-1')
    embedding_layer = tf.keras.layers.Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
    embed = embedding_layer(main_input)

    # 卷积核大小分别为2,3,4
    cnn1 = tf.keras.layers.Conv1D(128, 1, padding='valid', strides=1, activation='relu')(embed)
    cnn1 = tf.keras.layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 1 + 1)(cnn1)
    cnn2 = tf.keras.layers.Conv1D(128, 2, padding='valid', strides=1, activation='relu')(embed)
    cnn2 = tf.keras.layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 2 + 1)(cnn2)
    cnn3 = tf.keras.layers.Conv1D(128, 3, padding='valid', strides=1, activation='relu')(embed)
    cnn3 = tf.keras.layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 3 + 1)(cnn3)
    #cnn4 = tf.keras.layers.Conv1D(128, 4, padding='valid', strides=1, activation='relu')(embed)
    #cnn4 = tf.keras.layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 4 + 1)(cnn4)
    #cnn5 = tf.keras.layers.Conv1D(128, 5, padding='valid', strides=1, activation='relu')(embed)
    #cnn5 = tf.keras.layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 5 + 1)(cnn5)
    # 合并三个模型的输出向量
    cnn = tf.keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
    #cnn = tf.keras.layers.concatenate([cnn2, cnn3, cnn4], axis=-1)
    flat = tf.keras.layers.Flatten()(cnn)
    drop = tf.keras.layers.Dropout(0.1)(flat)
    main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output-1')(drop)
    model = tf.keras.models.Model(inputs=main_input, outputs=main_output)



    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    print(weights)
    print(type(weights))
    #weights = [0.5, 5]
    weights = dict(enumerate(weights))
    print(weights)
    print(type(weights))
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=4, class_weight=weights)
    model.save("model_result2/" + LANGUAGE_TYPE + "_textCNN/")

def TextCNN_model_3(x_train, y_train, x_val, y_val, embedding_matrix):
    main_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input-1')
    embedding_layer = tf.keras.layers.Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
    embed = embedding_layer(main_input)

    # 卷积核大小分别为2,3,4
    cnn1 = tf.keras.layers.Conv1D(128, 1, padding='valid', strides=1, activation='relu')(embed)
    cnn1 = tf.keras.layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 1 + 1)(cnn1)
    cnn2 = tf.keras.layers.Conv1D(128, 2, padding='valid', strides=1, activation='relu')(embed)
    cnn2 = tf.keras.layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 2 + 1)(cnn2)
    cnn3 = tf.keras.layers.Conv1D(128, 3, padding='valid', strides=1, activation='relu')(embed)
    cnn3 = tf.keras.layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 3 + 1)(cnn3)
    cnn4 = tf.keras.layers.Conv1D(128, 4, padding='valid', strides=1, activation='relu')(embed)
    cnn4 = tf.keras.layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 4 + 1)(cnn4)
    #cnn5 = tf.keras.layers.Conv1D(128, 5, padding='valid', strides=1, activation='relu')(embed)
    #cnn5 = tf.keras.layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 5 + 1)(cnn5)
    # 合并三个模型的输出向量
    cnn = tf.keras.layers.concatenate([cnn1, cnn2, cnn3, cnn4], axis=-1)
    #cnn = tf.keras.layers.concatenate([cnn2, cnn3, cnn4], axis=-1)
    flat = tf.keras.layers.Flatten()(cnn)
    drop = tf.keras.layers.Dropout(0.5)(flat)
    main_output = tf.keras.layers.Dense(3, activation='softmax', name='output-1')(drop)
    model = tf.keras.models.Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)
    model.save("model_result3/" + LANGUAGE_TYPE + "_textCNN/")


# In[36]:


if TASK_CLASS_NUMBER == 2:
    TextCNN_model_2(x_train, y_train, x_test, y_test, embedding_matrix)
else:
    TextCNN_model_3(x_train, y_train, x_test, y_test, embedding_matrix)


# In[37]:


if TASK_CLASS_NUMBER == 2:
    load_model = tf.keras.models.load_model("model_result2/" + LANGUAGE_TYPE + "_textCNN/")
else:
    load_model = tf.keras.models.load_model("model_result3/" + LANGUAGE_TYPE + "_textCNN/")


# In[38]:


if TASK_CLASS_NUMBER == 3:
    prob = load_model.predict(x_test)
    print(y_test[0])
    print(y_test_ori[0])
    print(prob[0][0])
    print(prob[0][1])
    print(prob[0][2])

    print(np.argmax(prob[0]))
    print(np.argmax(y_test[0]))


# In[39]:




# x_test, y_test
#  test_pos_number, test_neg_number
#x_test =x_train
#y_test = y_train

prob = load_model.predict(x_test)

if TASK_CLASS_NUMBER == 2:
    acc_number = 0
    predict_pos_number = 0
    predict_neg_number = 0
    acc_pos_number = 0
    acc_neg_number = 0

    error_pos_query = []
    error_pos_input = []
    error_pos_score = []
    for i in range(0, len(prob)):
        if prob[i][0] >= 0.5:
            predict_pos_number += 1
            if y_test[i] == 1:
                acc_pos_number += 1
                acc_number += 1
        else:
            predict_neg_number += 1
            if y_test[i] == 0:
                acc_neg_number += 1
                acc_number += 1
            #else:
                #error_pos_query.append(x_test_ori[i])
                #error_pos_input.append(x_test[i])
                #error_pos_score.append(prob[i][0])

    print("正样本个数:" + str(test_pos_number) + ", 预测正样本个数" + str(predict_pos_number) + ", 正确预测正样本个数:" + str(acc_pos_number) +
          ", 召回率:" + str(acc_pos_number/test_pos_number) + ", 准确率:" + str(acc_pos_number/predict_pos_number))
    print("负样本个数:" + str(test_neg_number) + ", 预测负样本个数" + str(predict_neg_number) + ", 正确预测负样本个数:" + str(acc_neg_number) +
          ", 召回率:" + str(acc_neg_number/test_neg_number) + ", 准确率:" + str(acc_neg_number/predict_neg_number))
    print("F1 = " + str(2 * acc_pos_number/test_pos_number * acc_pos_number/predict_pos_number / (acc_pos_number/predict_pos_number + acc_pos_number/predict_pos_number)))

    print("acc = "  + str(acc_number/(len(x_test))))

    #print("没有召回的正样本sample:")
    #for i in range(0, len(error_pos_query)):
    #    print("query: {}, input: {}, model score: {}".format(error_pos_query[i], error_pos_input[i], error_pos_score[i]))

else:
    # 多分类
    acc_number = 0
    predict_pos_2_number = 0
    predict_pos_1_number = 0
    predict_neg_number = 0
    acc_pos_2_number = 0
    acc_pos_1_number = 0
    acc_neg_number = 0

    predict_pos_number = 0
    acc_pos_number = 0

    error_pos_query = []
    error_pos_input = []
    error_pos_score = []
    for i in range(0, len(prob)):
        y_pred_class = np.argmax(prob[i])
        y_true_class = np.argmax(y_test[i])
        if y_pred_class == 2:
            predict_pos_2_number += 1
            predict_pos_number += 1
            if y_true_class == 2:
                acc_pos_2_number += 1
                acc_pos_number += 1
                acc_number += 1
            elif y_true_class == 1:
                acc_pos_number += 1
        elif y_pred_class == 1:
            predict_pos_1_number += 1
            predict_pos_number += 1
            if y_true_class == 1:
                acc_pos_1_number += 1
                acc_pos_number += 1
                acc_number += 1
            elif y_true_class == 2:
                acc_pos_number += 1
        else:
            predict_neg_number += 1
            if y_true_class == 0:
                acc_neg_number += 1
                acc_number += 1
            else:
                error_pos_query.append(x_test_ori[i])
                error_pos_input.append(x_test[i])
                error_pos_score.append(prob[i])


    print("2正样本个数:" + str(test_pos_2_number) + ", 预测正样本个数" + str(predict_pos_2_number) + ", 正确预测正样本个数:" + str(acc_pos_2_number) +
          ", 召回率:" + str(acc_pos_2_number/test_pos_2_number) + ", 准确率:" + str(acc_pos_2_number/predict_pos_2_number))

    print("1正样本个数:" + str(test_pos_1_number) + ", 预测正样本个数" + str(predict_pos_1_number) + ", 正确预测正样本个数:" + str(acc_pos_1_number) +
          ", 召回率:" + str(acc_pos_1_number/test_pos_1_number) + ", 准确率:" + str(acc_pos_1_number/predict_pos_1_number))

    print("负样本个数:" + str(test_neg_number) + ", 预测负样本个数" + str(predict_neg_number) + ", 正确预测负样本个数:" + str(acc_neg_number) +
          ", 召回率:" + str(acc_neg_number/test_neg_number) + ", 准确率:" + str(acc_neg_number/predict_neg_number))
    print("正样本个数:" + str(test_pos_1_number + test_pos_2_number) + ", 预测正样本个数" + str(predict_pos_number) + ", 正确预测正样本个数:" + str(acc_pos_number) +
          ", 召回率:" + str(acc_pos_number/(test_pos_1_number+test_pos_2_number)) + ", 准确率:" + str(acc_pos_number/predict_pos_number))



    print("acc = "  + str(acc_number/(len(x_test))))


# In[40]:


# 人工测试集合
dcg_query = []
dcg_labels = []

if TASK_CLASS_NUMBER == 2:
    f = open("../data/query/model_data/{}/{}_val_binary.txt".format(LANGUAGE_TYPE, LANGUAGE_TYPE))
else:
    f = open("../data/query/model_data/{}/seg/{}_val_multi_seg.txt".format(LANGUAGE_TYPE, LANGUAGE_TYPE))


dcg_pos_number = 0
dcg_pos_2_number = 0
dcg_pos_1_number = 0
dcg_neg_number = 0
n = 0
for line in f:
    values = line.strip().split("\t")
    if len(values) != 2:
        #print(line)
        continue
    label = (int)(values[1])
    if values[0].strip() in query_texts:
        n += 1
        continue
    if label == 2:
        dcg_pos_number += 1
        dcg_pos_2_number += 1
    elif label == 1:
        dcg_pos_number += 1
        dcg_pos_1_number += 1
    else:
        dcg_neg_number += 1
    dcg_query.append(values[0].strip())
    dcg_labels.append(label)

f.close()

print("filter number {}".format(n))


x_dcg_sequences = tokenizer.texts_to_sequences(dcg_query)
x_dcg = tf.keras.preprocessing.sequence.pad_sequences(x_dcg_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
if TASK_CLASS_NUMBER == 2:
    y_dcg = np.asarray(dcg_labels)
else:
    y_dcg = tf.keras.utils.to_categorical(dcg_labels)

prob = load_model.predict(x_dcg)


if TASK_CLASS_NUMBER == 2:
    acc_number = 0
    predict_pos_number = 0
    predict_neg_number = 0
    acc_pos_number = 0
    acc_neg_number = 0

    error_pos_query = []
    error_pos_input = []
    error_pos_score = []
    for i in range(0, len(prob)):
        if prob[i][0] >= 0.5:
            predict_pos_number += 1
            if dcg_labels[i] == 1:
                acc_pos_number += 1
                acc_number += 1
        else:
            predict_neg_number += 1
            if dcg_labels[i] == 0:
                acc_neg_number += 1
                acc_number += 1
            else:
                error_pos_query.append(dcg_query[i])
                error_pos_input.append(x_dcg[i])
                error_pos_score.append(prob[i][0])

    print("正样本个数:" + str(dcg_pos_number) + ", 预测正样本个数" + str(predict_pos_number) + ", 正确预测正样本个数:" + str(acc_pos_number) +
          ", 召回率:" + str(acc_pos_number/dcg_pos_number) + ", 准确率:" + str(acc_pos_number/predict_pos_number))
    print("负样本个数:" + str(dcg_neg_number) + ", 预测负样本个数" + str(predict_neg_number) + ", 正确预测负样本个数:" + str(acc_neg_number) +
          ", 召回率:" + str(acc_neg_number/dcg_neg_number) + ", 准确率:" + str(acc_neg_number/predict_neg_number))
    print("F1 = " + str(2 * acc_pos_number/dcg_pos_number * acc_pos_number/predict_pos_number / (acc_pos_number/predict_pos_number + acc_pos_number/dcg_pos_number)))

    print("acc = "  + str(acc_number/(len(dcg_query))))


    print("没有召回的正样本sample:")
    for i in range(0, len(error_pos_query)):
        print("query: {}, input: {}, model score: {}".format(error_pos_query[i], error_pos_input[i], error_pos_score[i]))
else:
    # 多分类
    acc_number = 0
    predict_pos_2_number = 0
    predict_pos_1_number = 0
    predict_neg_number = 0
    acc_pos_2_number = 0
    acc_pos_1_number = 0
    acc_neg_number = 0

    predict_pos_number = 0
    acc_pos_number = 0

    error_pos_query = []
    error_pos_input = []
    error_pos_score = []
    error_pos_true_score = []
    for i in range(0, len(prob)):
        y_pred_class = np.argmax(prob[i])
        y_true_class = np.argmax(y_dcg[i])
        if y_pred_class == 2:
            predict_pos_2_number += 1
            predict_pos_number += 1
            if y_true_class == 2:
                acc_pos_2_number += 1
                acc_pos_number += 1
                acc_number += 1
            elif y_true_class == 1:
                acc_pos_number += 1
        elif y_pred_class == 1:
            predict_pos_1_number += 1
            predict_pos_number += 1
            if y_true_class == 1:
                acc_pos_1_number += 1
                acc_pos_number += 1
                acc_number += 1
            elif y_true_class == 2:
                acc_pos_number += 1
        else:
            predict_neg_number += 1
            if y_true_class == 0:
                acc_neg_number += 1
                acc_number += 1
            else:
                error_pos_query.append(x_test_ori[i])
                error_pos_input.append(x_test[i])
                error_pos_score.append(prob[i])
                error_pos_true_score.append(y_true_class)


    print("2正样本个数:" + str(dcg_pos_2_number) + ", 预测正样本个数" + str(predict_pos_2_number) + ", 正确预测正样本个数:" + str(acc_pos_2_number) +
          ", 召回率:" + str(acc_pos_2_number/dcg_pos_2_number) + ", 准确率:" + str(acc_pos_2_number/predict_pos_2_number))

    print("1正样本个数:" + str(dcg_pos_1_number) + ", 预测正样本个数" + str(predict_pos_1_number) + ", 正确预测正样本个数:" + str(acc_pos_1_number) +
          ", 召回率:" + str(acc_pos_1_number/dcg_pos_1_number) + ", 准确率:" + str(acc_pos_1_number/predict_pos_1_number))

    print("负样本个数:" + str(dcg_neg_number) + ", 预测负样本个数" + str(predict_neg_number) + ", 正确预测负样本个数:" + str(acc_neg_number) +
          ", 召回率:" + str(acc_neg_number/dcg_neg_number) + ", 准确率:" + str(acc_neg_number/predict_neg_number))

    print("正样本个数:" + str(dcg_pos_2_number + dcg_pos_1_number) + ", 预测正样本个数" + str(predict_pos_number) + ", 正确预测正样本个数:" + str(acc_pos_number) +
          ", 召回率:" + str(acc_pos_number/(dcg_pos_2_number+dcg_pos_1_number)) + ", 准确率:" + str(acc_pos_number/predict_pos_number))


    print("acc = "  + str(acc_number/(len(dcg_query))))



    print("没有召回的正样本sample:")
    for i in range(0, len(error_pos_query)):
        print("query: {}, input: {}, model score: {}, true lable: {}".format(error_pos_query[i], error_pos_input[i], error_pos_score[i], error_pos_true_score[i]))


# In[41]:


# random query
dcg_query = []
dcg_labels = []

if TASK_CLASS_NUMBER == 2:
    f = open("../data/query/model_data/{}/seg/{}_random_query_seg.txt".format(LANGUAGE_TYPE, LANGUAGE_TYPE))
else:
    f = open("../data/query/model_data/{}/seg/{}_val_multi_seg.txt".format(LANGUAGE_TYPE, LANGUAGE_TYPE))


dcg_pos_number = 0
dcg_pos_2_number = 0
dcg_pos_1_number = 0
dcg_neg_number = 0
n = 0
for line in f:
    values = line.strip().split("\t")
    if len(values) > 2:
        #print(line)
        continue
    label = 0
    if len(values) == 2:
        label = 1
    if values[0].strip() in query_texts:
        n += 1
        continue
    if label == 2:
        dcg_pos_number += 1
        dcg_pos_2_number += 1
    elif label == 1:
        dcg_pos_number += 1
        dcg_pos_1_number += 1
    else:
        dcg_neg_number += 1
    dcg_query.append(values[0].strip())
    dcg_labels.append(label)

f.close()

print("filter number {}".format(n))


x_dcg_sequences = tokenizer.texts_to_sequences(dcg_query)
x_dcg = tf.keras.preprocessing.sequence.pad_sequences(x_dcg_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
if TASK_CLASS_NUMBER == 2:
    y_dcg = np.asarray(dcg_labels)
else:
    y_dcg = tf.keras.utils.to_categorical(dcg_labels)

prob = load_model.predict(x_dcg)

print("x_dcg type {}".format(type(x_dcg)))
print(x_dcg[0])

if TASK_CLASS_NUMBER == 2:
    acc_number = 0
    predict_pos_number = 0
    predict_neg_number = 0
    acc_pos_number = 0
    acc_neg_number = 0

    error_pos_query = []
    error_pos_input = []
    error_pos_score = []
    for i in range(0, len(prob)):
        if prob[i][0] >= 0.5:
            predict_pos_number += 1
            if dcg_labels[i] == 1:
                acc_pos_number += 1
                acc_number += 1
        else:
            predict_neg_number += 1
            if dcg_labels[i] == 0:
                acc_neg_number += 1
                acc_number += 1
            else:
                error_pos_query.append(dcg_query[i])
                error_pos_input.append(x_dcg[i])
                error_pos_score.append(prob[i][0])

    print("正样本个数:" + str(dcg_pos_number) + ", 预测正样本个数" + str(predict_pos_number) + ", 正确预测正样本个数:" + str(acc_pos_number) +
          ", 召回率:" + str(acc_pos_number/dcg_pos_number) + ", 准确率:" + str(acc_pos_number/predict_pos_number))
    print("负样本个数:" + str(dcg_neg_number) + ", 预测负样本个数" + str(predict_neg_number) + ", 正确预测负样本个数:" + str(acc_neg_number) +
          ", 召回率:" + str(acc_neg_number/dcg_neg_number) + ", 准确率:" + str(acc_neg_number/predict_neg_number))
    print("F1 = " + str(2 * acc_pos_number/dcg_pos_number * acc_pos_number/predict_pos_number / (acc_pos_number/predict_pos_number + acc_pos_number/dcg_pos_number)))

    print("acc = "  + str(acc_number/(len(dcg_query))))


    print("没有召回的正样本sample:")
    for i in range(0, len(error_pos_query)):
        print("query: {}, input: {}, model score: {}".format(error_pos_query[i], error_pos_input[i], error_pos_score[i]))
else:
    # 多分类
    acc_number = 0
    predict_pos_2_number = 0
    predict_pos_1_number = 0
    predict_neg_number = 0
    acc_pos_2_number = 0
    acc_pos_1_number = 0
    acc_neg_number = 0

    predict_pos_number = 0
    acc_pos_number = 0

    error_pos_query = []
    error_pos_input = []
    error_pos_score = []
    error_pos_true_score = []
    for i in range(0, len(prob)):
        y_pred_class = np.argmax(prob[i])
        y_true_class = np.argmax(y_dcg[i])
        if y_pred_class == 2:
            predict_pos_2_number += 1
            predict_pos_number += 1
            if y_true_class == 2:
                acc_pos_2_number += 1
                acc_pos_number += 1
                acc_number += 1
            elif y_true_class == 1:
                acc_pos_number += 1
        elif y_pred_class == 1:
            predict_pos_1_number += 1
            predict_pos_number += 1
            if y_true_class == 1:
                acc_pos_1_number += 1
                acc_pos_number += 1
                acc_number += 1
            elif y_true_class == 2:
                acc_pos_number += 1
        else:
            predict_neg_number += 1
            if y_true_class == 0:
                acc_neg_number += 1
                acc_number += 1
            else:
                error_pos_query.append(x_test_ori[i])
                error_pos_input.append(x_test[i])
                error_pos_score.append(prob[i])
                error_pos_true_score.append(y_true_class)


    print("2正样本个数:" + str(dcg_pos_2_number) + ", 预测正样本个数" + str(predict_pos_2_number) + ", 正确预测正样本个数:" + str(acc_pos_2_number) +
          ", 召回率:" + str(acc_pos_2_number/dcg_pos_2_number) + ", 准确率:" + str(acc_pos_2_number/predict_pos_2_number))

    print("1正样本个数:" + str(dcg_pos_1_number) + ", 预测正样本个数" + str(predict_pos_1_number) + ", 正确预测正样本个数:" + str(acc_pos_1_number) +
          ", 召回率:" + str(acc_pos_1_number/dcg_pos_1_number) + ", 准确率:" + str(acc_pos_1_number/predict_pos_1_number))

    print("负样本个数:" + str(dcg_neg_number) + ", 预测负样本个数" + str(predict_neg_number) + ", 正确预测负样本个数:" + str(acc_neg_number) +
          ", 召回率:" + str(acc_neg_number/dcg_neg_number) + ", 准确率:" + str(acc_neg_number/predict_neg_number))

    print("正样本个数:" + str(dcg_pos_2_number + dcg_pos_1_number) + ", 预测正样本个数" + str(predict_pos_number) + ", 正确预测正样本个数:" + str(acc_pos_number) +
          ", 召回率:" + str(acc_pos_number/(dcg_pos_2_number+dcg_pos_1_number)) + ", 准确率:" + str(acc_pos_number/predict_pos_number))


    print("acc = "  + str(acc_number/(len(dcg_query))))



    print("没有召回的正样本sample:")
    for i in range(0, len(error_pos_query)):
        print("query: {}, input: {}, model score: {}, true lable: {}".format(error_pos_query[i], error_pos_input[i], error_pos_score[i], error_pos_true_score[i]))
