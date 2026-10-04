# ============================================================
# export_tflite.py — converts trained model to TensorFlow Lite
# ============================================================
# export_tflite.py — converts trained model to TensorFlow Lite and prints input details

import tensorflow as tf
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = "deep_session_multimodel.h5"
OUTPUT_PATH = "deep_session_multimodel.tflite"

print(f"Loading trained model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -------------------------
# Print Keras model inputs
# -------------------------
print("Keras model inputs:")
for i, layer_input in enumerate(model.inputs):
    # layer_input.shape is a tuple; convert to list safely
    shape = list(layer_input.shape)
    # replace None (batch) with -1 for display consistency
    shape_display = [dim if dim is not None else -1 for dim in shape]
    dtype = getattr(layer_input, "dtype", None)
    print(f"  Input[{i}] name={layer_input.name} shape={shape_display} dtype={dtype}")

# Compute flattened element count per example for the first input
first_input = model.inputs[0]
shape_list = list(first_input.shape)  # e.g. [None, 66]
element_dims = shape_list[1:] if len(shape_list) > 1 else shape_list
num_elements = 1
for d in element_dims:
    num_elements *= (d if d is not None else 1)
print(f"Flattened elements (per example): {num_elements}")
sample = [0.0] * int(num_elements)
print("Sample input vector (first 20 values):", sample[:20])
print("Sample input vector length:", len(sample))

# -------------------------
# Convert to TFLite
# -------------------------
print("Converting model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.experimental_new_converter = True
tflite_model = converter.convert()

# -------------------------
# Save TFLite model
# -------------------------
print(f"Saving TFLite model to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)
print("Export complete. File saved as deep_session_multimodel.tflite")

# -------------------------
# Inspect TFLite model inputs
# -------------------------
tflite_path = OUTPUT_PATH
interpreter = Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print("TFLite model input details:")
for i, inp in enumerate(input_details):
    shape = list(map(int, inp["shape"].tolist()))
    dtype = inp["dtype"]                     # numpy dtype object
    quantization = inp.get("quantization", None)
    num_elements = int(np.prod(shape))
    bytes_per_element = np.dtype(dtype).itemsize
    expected_bytes = num_elements * bytes_per_element
    print(f"  Input[{i}] name={inp.get('name')} shape={shape} dtype={dtype} num_elements={num_elements} expected_bytes={expected_bytes} quant={quantization}")

# Build and print a sample flattened input vector matching the interpreter's expected element count
num_elements = int(np.prod(input_details[0]["shape"]))
sample_vector = np.zeros(num_elements, dtype=np.float32)
print("Sample flattened input vector length:", sample_vector.size)
print("Sample flattened input vector head:", sample_vector[:20])

