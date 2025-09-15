import cv2
import numpy as np
from tensorflow import keras

# 1. 載入模型
model = keras.models.load_model("mnist_model.h5")

# 2. 讀取圖片 (灰階)
img = cv2.imread("mydigit.png", cv2.IMREAD_GRAYSCALE)

# 3. 預處理
img = cv2.resize(img, (28, 28))     # MNIST 輸入大小
img = 255 - img                     # 反相：黑字白底 → 白字黑底
img = img.astype("float32") / 255.0
img = img.reshape(1, 28, 28, 1)     # (1, 28, 28, 1)

# 4. 預測
prediction = model.predict(img)
digit = np.argmax(prediction)

print("模型預測結果:", digit)
