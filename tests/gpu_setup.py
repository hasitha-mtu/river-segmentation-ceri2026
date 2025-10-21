import tensorflow as tf

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(f"physical_devices : {physical_devices}")
    print(tf.__version__)
