import tensorflow as tf

# モデルを変換
converter = tf.lite.TFLiteConverter.from_saved_model(
    "./models2/save_models/98/1107/SavedModel"
    )
tflite_model = converter.convert()

with open("./models2/save_models/98/1107/pfld_98.tflite", 'wb') as f:
    f.write(tflite_model)
