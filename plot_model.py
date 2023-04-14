import tensorflow as tf

model = tf.keras.models.load_model("model_result2/br_textCNN/")
model.summary(line_length = 200, positions = [.22, .55, .67, 1.])
tf.keras.utils.plot_model(model, "textCNN.png", show_shapes=True)
