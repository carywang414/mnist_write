import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 1. 載入 MNIST 資料集
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. 正規化 & 調整維度
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)   # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)     # (10000, 28, 28, 1)

# 3. One-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 4. 建立模型
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

# 5. 編譯模型
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 6. 訓練模型
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_split=0.1
)

# 7. 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("測試集準確率:", test_acc)

# 8. 存模型
model.save("mnist_model.h5")
print("模型已儲存為 mnist_model.h5")


# # 9. 可視化訓練過程
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='test')
# plt.legend()
# plt.show()
