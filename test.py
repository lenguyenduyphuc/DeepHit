import tensorflow as tf
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)
# Or in TF 1.x:
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())