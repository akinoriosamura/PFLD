import tensorflow as tf

# Load the saved keras model back.
k_model = tf.keras.models.load_model(
    "SavedModelPre",
    custom_objects=None,
    compile=True
)
# k_model = tf.keras.experimental.load_from_saved_model("SavedModelPre")
k_model.summary()

k_model.save('model.h5', include_optimizer=False)

converter = tf.lite.TFLiteConverter.from_keras_model_file("model.h5")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
