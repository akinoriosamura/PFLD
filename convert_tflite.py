import tensorflow as tf

save_model = "./models2/save_models/68/PFLD_growing_68_pre_WFLW"

# モデルを変換
converter = tf.lite.TFLiteConverter.from_saved_model(
    save_model + "/SavedModel"
    )
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

with open(save_model + "/PFLD_OPTIMIZE_FOR_SIZE_growing_pre_WFLW.tflite", 'wb') as f:
    f.write(tflite_model)
