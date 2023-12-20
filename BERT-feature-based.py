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


device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
#tokenizer = tokenizer.to(device)

bert_model = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
bert_model = bert_model.to(device)

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

origin_train_data = CWD + "br_train_binary.txt.full.v2"
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

EMBEDDING_DIM = 100

x_train_ori, x_test_ori, y_train_ori, y_test_ori = train_test_split(query_texts, query_labels, test_size=0.1)


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

torch.cuda.empty_cache()

def get_input_vector(query):
    encoded_input = tokenizer(query, return_tensors='pt', padding='max_length', max_length=50)
    encoded_input = encoded_input.to(device)
    with torch.no_grad():
        output = bert_model(**encoded_input)
    ret = tf.convert_to_tensor(output.last_hidden_state[:, 0].tolist())
    return ret

def parse(s, label):
  return get_input_vector([x.numpy().decode('utf-8') for x in s]), label

batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_ori, y_train_ori))
train_data = train_dataset.batch(batch_size).map(lambda x,y:tf.py_function(parse, [x, y], (tf.float32, tf.int32)))

print(len(x_test_ori))
print(len(y_test_ori))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test_ori, y_test_ori))
test_data = test_dataset.batch(batch_size).map(lambda x,y:tf.py_function(parse, [x, y], (tf.float32, tf.int32)))

if TASK_CLASS_NUMBER == 2:
    y_train = np.asarray(y_train_ori, dtype=np.int32)
    y_test = np.asarray(y_test_ori, dtype=np.int32)
else:
    y_train = tf.keras.utils.to_categorical(y_train_ori)
    y_test = tf.keras.utils.to_categorical(y_test_ori)

def custom_loss(y_true, y_pred):
    if y_true == 1:
      ret = - (0.8 * tf.math.log(y_pred) + 0.2 * tf.math.log(1 - y_pred))
    else:
      ret = - (0.2 * tf.math.log(y_pred) + 0.8 * tf.math.log(1 - y_pred))
    return ret

def backend_reshape_restore_input(x):
    return tf.keras.backend.reshape(x, (-1, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM + LSTM_DIM * 4))

def backend_reshape_reduce_input(x):
    return tf.keras.backend.reshape(x, (-1, EMBEDDING_DIM))

def backend_reshape_input(x):
    return tf.keras.backend.reshape(x, (-1, MAX_WORD_LENGTH, EMBEDDING_CHAR_DIM))

def backend_reshape(x):
    return tf.keras.backend.reshape(x, (-1, MAX_SEQUENCE_LENGTH, LSTM_DIM * 2))

def switch_layer(inputs):
    inp, emb = inputs
    zeros = tf.zeros_like(inp, dtype = tf.float32)
    ones = tf.ones_like(inp, dtype = tf.float32)
    inp = tf.keras.backend.switch(tf.keras.backend.greater(inp, 0), ones, zeros)
    inp = tf.expand_dims(inp, -1)
    return inp * emb

def TextCNN_model_2(train, val):
    main_input = tf.keras.layers.Input(shape=(768), dtype='float', name='input_1')
    main_output = tf.keras.layers.Dense(48, activation='relu', name='output_1')(main_input)
    main_output = tf.keras.layers.LayerNormalization(axis=[-1])(main_output)
    drop = tf.keras.layers.Dropout(0.1)(main_output)
    main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output_2')(drop)
    model = tf.keras.models.Model(inputs=main_input, outputs=main_output)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), optimizer='Nadam', metrics=['binary_crossentropy'])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      CWD + "model_result2_best/" + LANGUAGE_TYPE + "_textCNN/",
      monitor = 'val_loss',
      save_best_only = True,
      save_weights_only = True)

    model.fit(train, validation_data=val, epochs=10, callbacks=[model_checkpoint_callback])
    model.save(CWD + "model_result2/" + LANGUAGE_TYPE + "_textCNN/")
    return model

def TextCNN_model_3(x_train, y_train, x_val, y_val, embedding_matrix):
    main_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input_1')
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
    main_output = tf.keras.layers.Dense(3, activation='softmax', name='output_1')(drop)
    model = tf.keras.models.Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    #model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)
    #model.save("model_result3/" + LANGUAGE_TYPE + "_textCNN/")

if TASK_CLASS_NUMBER == 2:
    load_model = TextCNN_model_2(train_data, test_data)
else:
    TextCNN_model_3(x_train, y_train, x_test, y_test)

if TASK_CLASS_NUMBER == 2:
    #load_model.load_weights(CWD + "model_result2_best/" + LANGUAGE_TYPE + "_textCNN/")
    load_model = tf.keras.models.load_model(CWD + "model_result2/" + LANGUAGE_TYPE + "_textCNN/")
    load_model.summary(line_length = 200, positions = [.22, .55, .67, 1.])
else:
    load_model = tf.keras.models.load_model("model_result3/" + LANGUAGE_TYPE + "_textCNN/")

if TASK_CLASS_NUMBER == 3:
    prob = load_model.predict(x_test)
    print(y_test[0])
    print(y_test_ori[0])
    print(prob[0][0])
    print(prob[0][1])
    print(prob[0][2])

    print(np.argmax(prob[0]))
    print(np.argmax(y_test[0]))

prob = load_model.predict(test_data)

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

    print("acc = "  + str(acc_number/(len(y_test))))

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

# 人工测试集合
dcg_query = []
dcg_labels = []

if TASK_CLASS_NUMBER == 2:
    f = open("{}_val_binary.txt".format(LANGUAGE_TYPE))
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

if TASK_CLASS_NUMBER == 2:
    y_dcg = np.asarray(dcg_labels, dtype=np.int32)
else:
    y_dcg = tf.keras.utils.to_categorical(dcg_labels)

dcg_dataset = tf.data.Dataset.from_tensor_slices((dcg_query, y_dcg))
dcg_dataset = dcg_dataset.batch(batch_size).map(lambda x,y:tf.py_function(parse, [x, y], (tf.float32, tf.int32)))
prob = load_model.predict(dcg_dataset)

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
                error_pos_score.append(prob[i][0])

    print("正样本个数:" + str(dcg_pos_number) + ", 预测正样本个数" + str(predict_pos_number) + ", 正确预测正样本个数:" + str(acc_pos_number) +
          ", 召回率:" + str(acc_pos_number/dcg_pos_number) + ", 准确率:" + str(acc_pos_number/predict_pos_number))
    print("负样本个数:" + str(dcg_neg_number) + ", 预测负样本个数" + str(predict_neg_number) + ", 正确预测负样本个数:" + str(acc_neg_number) +
          ", 召回率:" + str(acc_neg_number/dcg_neg_number) + ", 准确率:" + str(acc_neg_number/predict_neg_number))
    print("F1 = " + str(2 * acc_pos_number/dcg_pos_number * acc_pos_number/predict_pos_number / (acc_pos_number/predict_pos_number + acc_pos_number/dcg_pos_number)))

    print("acc = "  + str(acc_number/(len(dcg_query))))


    print("没有召回的正样本sample:")
    for i in range(0, len(error_pos_query)):
        print("query: {}, model score: {}".format(error_pos_query[i], error_pos_score[i]))
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
        print("query: {}, model score: {}, true lable: {}".format(error_pos_query[i], error_pos_score[i], error_pos_true_score[i]))



# random query
dcg_query = []
dcg_labels = []

if TASK_CLASS_NUMBER == 2:
    f = open("{}_random_query_seg.txt".format(LANGUAGE_TYPE))
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
    #if values[0].strip() in query_texts:
        #n += 1
        #continue
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


if TASK_CLASS_NUMBER == 2:
    y_dcg = np.asarray(dcg_labels, dtype=np.int32)
else:
    y_dcg = tf.keras.utils.to_categorical(dcg_labels)


dcg_dataset = tf.data.Dataset.from_tensor_slices((dcg_query, y_dcg))
dcg_dataset = dcg_dataset.batch(batch_size).map(lambda x,y:tf.py_function(parse, [x, y], (tf.float32, tf.int32)))
prob = load_model.predict(dcg_dataset)


if TASK_CLASS_NUMBER == 2:
    acc_number = 0
    predict_pos_number = 0
    predict_neg_number = 0
    acc_pos_number = 0
    acc_neg_number = 0

    error_pos_query = []
    error_pos_orig_input = []
    error_pos_input = []
    error_pos_score = []
    for i in range(0, len(prob)):
        if prob[i][0] >= 0.40:
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
                error_pos_score.append(prob[i][0])

    for i in range(0, 61):
      print("====" + str(i) + "======")
      print(dcg_query[i])
      print(prob[i][0])

    print(dcg_query[40])
    print(prob[40][0])


    print("没有召回的正样本sample:")
    for i in range(0, len(error_pos_query)):
        print("query: {}, model score: {}".format(error_pos_query[i], error_pos_score[i]))

    print("正样本个数:" + str(dcg_pos_number) + ", 预测正样本个数" + str(predict_pos_number) + ", 正确预测正样本个数:" + str(acc_pos_number) +
          ", 召回率:" + str(acc_pos_number/dcg_pos_number) + ", 准确率:" + str(acc_pos_number/predict_pos_number))
    print("负样本个数:" + str(dcg_neg_number) + ", 预测负样本个数" + str(predict_neg_number) + ", 正确预测负样本个数:" + str(acc_neg_number) +
          ", 召回率:" + str(acc_neg_number/dcg_neg_number) + ", 准确率:" + str(acc_neg_number/predict_neg_number))
    print("F1 = " + str(2 * acc_pos_number/dcg_pos_number * acc_pos_number/predict_pos_number / (acc_pos_number/predict_pos_number + acc_pos_number/dcg_pos_number)))

    print("acc = "  + str(acc_number/(len(dcg_query))))


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
