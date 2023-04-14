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

LANGUAGE_TYPE = "in"
TASK_CLASS_NUMBER = 2  # 2代表二分类， 3 代表多分类
MAX_SEQUENCE_LENGTH = 16
MAX_WORD_LENGTH = 10
POS_SAMPLE_WEIGHT = 1  # 正样本重复倍数
EMBEDDING_CHAR_DIM = 25
LSTM_DIM = 25


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





tokenizer=tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(query_texts)
word_index=tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

tokenizer_char=tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer_char.fit_on_texts(query_texts)
char_index=tokenizer_char.word_index
print('Found %s unique tokens.' % len(char_index))

bw_result = open('i18_q_sex_classify_' + LANGUAGE_TYPE + '_' + str(TASK_CLASS_NUMBER) + '_char_index_dict.txt', 'w')
for char, i in char_index.items():
    bw_result.write(char + "\t" + str(i) + '\n')
bw_result.flush()
bw_result.close()

# 打印提取的样本
f = open(LANGUAGE_TYPE + '_' + str(TASK_CLASS_NUMBER) + '_use_data.txt', 'w')
for i in range(0, len(query_texts)):
    f.write(query_texts[i] + '\t' + str(query_labels[i]) + '\n')
f.flush()
f.close()
print("OK")

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


x_train_sequences = tokenizer.texts_to_sequences(x_train_ori)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')

x_train_word_sequences = [text.split() for text in x_train_ori]
print("x_train_word_sequences[0] is:")
print(x_train_word_sequences[0])
x_train_word_sequences = tf.keras.preprocessing.sequence.pad_sequences(x_train_word_sequences, dtype=object, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', value='')
x_train_char_sequences = [tokenizer_char.texts_to_sequences(word_seq) for word_seq in x_train_word_sequences]
x_train_char_sequences = [tf.keras.preprocessing.sequence.pad_sequences(char_seq, maxlen=MAX_WORD_LENGTH, padding='pre') for char_seq in x_train_char_sequences]

x_test_sequences = tokenizer.texts_to_sequences(x_test_ori)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')

x_test_word_sequences = [text.split() for text in x_test_ori]
x_test_word_sequences = tf.keras.preprocessing.sequence.pad_sequences(x_test_word_sequences, dtype=object, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', value='')
x_test_char_sequences = [tokenizer_char.texts_to_sequences(word_seq) for word_seq in x_test_word_sequences]
x_test_char_sequences = [tf.keras.preprocessing.sequence.pad_sequences(char_seq, maxlen=MAX_WORD_LENGTH, padding='pre') for char_seq in x_test_char_sequences]


if TASK_CLASS_NUMBER == 2:
    y_train = np.asarray(y_train_ori)
    y_test = np.asarray(y_test_ori)
else:
    y_train = tf.keras.utils.to_categorical(y_train_ori)
    y_test = tf.keras.utils.to_categorical(y_test_ori)

print(x_train_ori[0])
print(x_train[0])
print(x_train_word_sequences[0])
print(x_train_char_sequences[0])
print(y_train[0])

print(x_train_ori[-2])
print(x_train[-2])
print(x_train_word_sequences[-2])
print(x_train_char_sequences[-2])
print(y_train[-2])

print(x_test_ori[0])
print(x_test[0])
print(x_test_word_sequences[0])
print(x_test_char_sequences[0])
print(y_test[0])


# x_train y_train
print(Counter(y_train))

#x_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
#print(Counter(y_resampled))

#x_train = x_resampled
#y_train = y_resampled






print(x_train[-10:-1])
print(y_train[-10:-1])
print(x_train_char_sequences[-2])

def backend_reshape_input(x):
    return tf.keras.backend.reshape(x, (-1, MAX_WORD_LENGTH, EMBEDDING_CHAR_DIM))

def backend_reshape(x):
    return tf.keras.backend.reshape(x, (-1, MAX_SEQUENCE_LENGTH, LSTM_DIM * 2))

def switch_layer(inputs):
    inp, emb = inputs
    zeros = tf.zeros_like(inp, dtype = tf.float32)
    ones = tf.ones_like(inp, dtype = tf.float32)
    inp = tf.keras.backend.switch(inp > 0, ones, zeros)
    inp = tf.expand_dims(inp, -1)
    return inp * emb

def TextCNN_model_2(x_train, x_train_char, y_train, x_val, x_val_char, y_val, embedding_matrix):
    main_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='input-1')
    embedding_layer = tf.keras.layers.Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
    embed = embedding_layer(main_input)

    main_char_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH, MAX_WORD_LENGTH), dtype='int32', name='input-2')
    embedding_char_layer = tf.keras.layers.Embedding(len(char_index) + 1,
                            EMBEDDING_CHAR_DIM,
                            input_length=MAX_WORD_LENGTH,
                            trainable=True)
    embed_char = embedding_char_layer(main_char_input)
    print(embedding_char_layer.output_shape)
    embed_char = tf.keras.layers.Lambda(switch_layer)([main_char_input, embed_char])
    print("embed_char.shape is:")
    print(embed_char.shape)
    print(embed_char[0].shape)
    embed_char = tf.keras.layers.Lambda(backend_reshape_input)(embed_char)
    print("embed_char.shape is:")
    print(embed_char.shape)

    #print(embed_char[0]._keras_mask)

    char_lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
          LSTM_DIM,
          input_shape=(MAX_WORD_LENGTH, EMBEDDING_CHAR_DIM),
          return_sequences=False,
          return_state=True),
        backward_layer = tf.keras.layers.LSTM(
          LSTM_DIM,
          input_shape=(MAX_WORD_LENGTH, EMBEDDING_CHAR_DIM),
          return_sequences=False,
          go_backwards=True,
          return_state=True),
        name='Bi-LSTM_Char')

    masking_layer = tf.keras.layers.Masking(mask_value=0., input_shape=(MAX_WORD_LENGTH, EMBEDDING_CHAR_DIM))
    embed_char = masking_layer(embed_char)
    print(embed_char._keras_mask)
    #embed_char = tf.keras.layers.LayerNormalization()(embed_char)
    char_lstm_output, _, cell_1, _, cell_2 = char_lstm_layer(embed_char)
    print(char_lstm_layer.output_shape)
    print(char_lstm_output.shape)

    char_lstm_output = tf.keras.layers.Lambda(backend_reshape)(char_lstm_output)
    cell_lstm_output = tf.keras.layers.concatenate([cell_1, cell_2], axis=-1)
    cell_lstm_output = tf.keras.layers.Lambda(backend_reshape)(cell_lstm_output)
    char_lstm_output = tf.keras.layers.concatenate([char_lstm_output, cell_lstm_output], axis=-1)

    embed = tf.keras.layers.concatenate([embed, char_lstm_output], axis=-1)
    embed = tf.keras.layers.Dropout(0.3)(embed)

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
    drop = tf.keras.layers.Dropout(0.3)(flat)
    main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='output-1')(drop)
    model = tf.keras.models.Model(inputs=[main_input, main_char_input], outputs=main_output)



    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    print(weights)
    print(type(weights))
    #weights = [0.5, 5]
    weights = dict(enumerate(weights))
    print(weights)
    print(type(weights))

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      "model_result2_best/" + LANGUAGE_TYPE + "_textCNN/",
      monitor = 'val_loss',
      save_best_only = True,
      save_weights_only = True)

    model.fit([x_train, x_train_char], y_train, validation_data=([x_val, x_val_char], y_val), epochs=20, batch_size=1, class_weight=weights, callbacks=[model_checkpoint_callback])
    model.save("model_result2/" + LANGUAGE_TYPE + "_textCNN/" + LANGUAGE_TYPE + ".h5", save_format="h5")
    return model

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

if TASK_CLASS_NUMBER == 2:
    print(x_train.shape)
    print(tf.convert_to_tensor(x_train_char_sequences).shape)
    load_model = TextCNN_model_2(x_train, tf.convert_to_tensor(x_train_char_sequences), y_train, x_test, tf.convert_to_tensor(x_test_char_sequences), y_test, embedding_matrix)
else:
    TextCNN_model_3(x_train, y_train, x_test, y_test, embedding_matrix)

if TASK_CLASS_NUMBER == 2:
    load_model.load_weights("model_result2_best/" + LANGUAGE_TYPE + "_textCNN/")
    #load_model = tf.keras.models.load_model("model_result2/" + LANGUAGE_TYPE + "_textCNN/" + LANGUAGE_TYPE + ".h5")
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



# x_test, y_test
#  test_pos_number, test_neg_number
#x_test =x_train
#y_test = y_train

prob = load_model.predict([x_test, tf.convert_to_tensor(x_test_char_sequences)], batch_size = 1)

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

x_dcg_word_sequences = [text.split() for text in dcg_query]
x_dcg_word_sequences = tf.keras.preprocessing.sequence.pad_sequences(x_dcg_word_sequences, dtype=object, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', value='')
x_dcg_char_sequences = [tokenizer_char.texts_to_sequences(word_seq) for word_seq in x_dcg_word_sequences]
x_dcg_char_sequences = [tf.keras.preprocessing.sequence.pad_sequences(char_seq, maxlen=MAX_WORD_LENGTH, padding='pre') for char_seq in x_dcg_char_sequences]


if TASK_CLASS_NUMBER == 2:
    y_dcg = np.asarray(dcg_labels)
else:
    y_dcg = tf.keras.utils.to_categorical(dcg_labels)

prob = load_model.predict([x_dcg, tf.convert_to_tensor(x_dcg_char_sequences)], batch_size = 1)


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

x_dcg_word_sequences = [text.split() for text in dcg_query]
x_dcg_word_sequences = tf.keras.preprocessing.sequence.pad_sequences(x_dcg_word_sequences, dtype=object, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', value='')
x_dcg_char_sequences = [tokenizer_char.texts_to_sequences(word_seq) for word_seq in x_dcg_word_sequences]
x_dcg_char_sequences = [tf.keras.preprocessing.sequence.pad_sequences(char_seq, maxlen=MAX_WORD_LENGTH, padding='pre') for char_seq in x_dcg_char_sequences]


prob = load_model.predict([x_dcg, tf.convert_to_tensor(x_dcg_char_sequences)], batch_size=1)

print("x_dcg type {}".format(type(x_dcg)))
print(x_dcg[0])

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
        if prob[i][0] >= 0.4:
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
                error_pos_orig_input.append(x_dcg_sequences[i])
                error_pos_input.append(x_dcg[i])
                error_pos_score.append(prob[i][0])


    #x vedeo ponro grafia ao vivo
    print("x vedeo ponro grafia ao vivo")
    print(dcg_query[0])
    print(x_dcg_sequences[0])
    print(x_dcg[0])
    print(prob[0][0])

    #xvideo ponro grafia ao vivo
    print("xvideo ponro grafia ao vivo")
    print(dcg_query[1])
    print(x_dcg_sequences[1])
    print(x_dcg[1])
    print(prob[1][0])

    #video por ono grafico
    print("video por ono grafico")
    print(dcg_query[2])
    print(x_dcg_sequences[2])
    print(x_dcg[2])
    print(prob[2][0])

    #🍑 🔞 🥒 bom dia
    print("🍑 🔞 🥒 bom dia")
    print(dcg_query[3])
    print(x_dcg_sequences[3])
    print(x_dcg[3])
    print(prob[3][0])

    #b ҈ o ҈ m ҈ d ҈ i ҈ a ҈
    print("b ҈ o ҈ m ҈ d ҈ i ҈ a ҈")
    print(dcg_query[4])
    print(x_dcg_sequences[4])
    print(x_dcg[4])
    print(prob[4][0])

    #gostozonas do kawai 19 anos
    print("gostozonas do kawai 19 anos")
    print(dcg_query[5])
    print(x_dcg_sequences[5])
    print(x_dcg[5])
    print(prob[5][0])

    #mulher dancando pedalada de
    print("mulher dancando pedalada de")
    print(dcg_query[6])
    print(x_dcg_sequences[6])
    print(x_dcg[6])
    print(prob[6][0])

    #mulher dancando pedalada de .
    print("mulher dancando pedalada de .")
    print(dcg_query[7])
    print(x_dcg_sequences[7])
    print(x_dcg[7])
    print(prob[7][0])

    #gostozonas mais do kawai
    print(dcg_query[8])
    print(x_dcg_sequences[8])
    print(x_dcg[8])
    print(prob[8][0])

    print(dcg_query[9])
    print(x_dcg_sequences[9])
    print(x_dcg[9])
    print(prob[9][0])

    print(dcg_query[10])
    print(x_dcg_sequences[10])
    print(x_dcg[10])
    print(prob[10][0])

    print(dcg_query[11])
    print(x_dcg_sequences[11])
    print(x_dcg[11])
    print(prob[11][0])

    for i in range(12, 41):
      print("====" + str(i) + "======")
      print(dcg_query[i])
      print(x_dcg_sequences[i])
      print(x_dcg[i])
      print(prob[i][0])

    print("正样本个数:" + str(dcg_pos_number) + ", 预测正样本个数" + str(predict_pos_number) + ", 正确预测正样本个数:" + str(acc_pos_number) +
          ", 召回率:" + str(acc_pos_number/dcg_pos_number) + ", 准确率:" + str(acc_pos_number/predict_pos_number))
    print("负样本个数:" + str(dcg_neg_number) + ", 预测负样本个数" + str(predict_neg_number) + ", 正确预测负样本个数:" + str(acc_neg_number) +
          ", 召回率:" + str(acc_neg_number/dcg_neg_number) + ", 准确率:" + str(acc_neg_number/predict_neg_number))
    print("F1 = " + str(2 * acc_pos_number/dcg_pos_number * acc_pos_number/predict_pos_number / (acc_pos_number/predict_pos_number + acc_pos_number/dcg_pos_number)))

    print("acc = "  + str(acc_number/(len(dcg_query))))


    print("没有召回的正样本sample:")
    for i in range(0, len(error_pos_query)):
        print("query: {}, orig_input: {}, input: {}, model score: {}".format(error_pos_query[i], error_pos_orig_input[i], error_pos_input[i], error_pos_score[i]))
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

