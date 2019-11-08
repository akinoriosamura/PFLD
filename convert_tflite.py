import tensorflow as tf

# モデルを変換
converter = tf.lite.TFLiteConverter.from_saved_model(
    "./SavedModel"
    )
tflite_model = converter.convert()

with open("./sample.tflite", 'wb') as f:
    f.write(tflite_model)
