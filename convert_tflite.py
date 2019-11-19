import tensorflow as tf

save_model = "./models2/save_models/pcn_growing_68/1113"

# モデルを変換
converter = tf.lite.TFLiteConverter.from_saved_model(
    save_model + "/SavedModel"
    )
tflite_model = converter.convert()

with open(save_model + "/PFLD_WFLW_98.tflite", 'wb') as f:
    f.write(tflite_model)
