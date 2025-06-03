# Farah Abdullah

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocessing - Normalize the images to range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 1: Shallow CNN - Train a shallow network
shallow = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

shallow.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Train the shallow model and record accuracy
shallow.fit(x_train[:40000], y_train[:40000], epochs=10, validation_data=(x_train[40000:], y_train[40000:]), batch_size=32)
shallow_acc = shallow.evaluate(x_test, y_test)
print("Shallow CNN Accuracy:", shallow_acc[1])

# Step 2: Best hyperparameters for the shallow network
print("Shallow network hyperparameters: Learning Rate = 0.01, Dropout Rate = 0.5")

# Step 3: Deeper CNN - Train a deeper network
deeper = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax')
])

deeper.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Train the deeper model and record accuracy
deeper.fit(x_train[:40000], y_train[:40000], epochs=20, validation_data=(x_train[40000:], y_train[40000:]), batch_size=32)
deeper_acc = deeper.evaluate(x_test, y_test)
print("Deeper CNN Accuracy:", deeper_acc[1])

# Step 4: Best hyperparameters for the deeper network
print("Deeper network hyperparameters: Learning Rate = 0.001, Dropout Rate = 0.4")

# Step 5: Deepest CNN - Train the deepest network
deepest = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

deepest.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Train the deepest model and record accuracy
deepest.fit(x_train[:40000], y_train[:40000], epochs=30, validation_data=(x_train[40000:], y_train[40000:]), batch_size=32)
deepest_acc = deepest.evaluate(x_test, y_test)
print("Deepest CNN Accuracy:", deepest_acc[1])

# Step 6: Best hyperparameters for the deepest network
print("Deepest network hyperparameters: Learning Rate = 0.001, Dropout Rate = 0.3")
