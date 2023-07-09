import tensorflow as tf


def my_activation(x):
    return tf.sin(x) 


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation=my_activation, input_shape=(28, 28, 1)),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
