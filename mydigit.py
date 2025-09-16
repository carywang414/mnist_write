import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# 載入模型
model = tf.keras.models.load_model("mnist_model.h5")

# 視窗
root = tk.Tk()
root.title("MNIST 手寫數字辨識")

# 畫布設定
canvas_size = 280  # 放大一點方便畫
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
canvas.grid(row=0, column=0, columnspan=4)

# PIL 畫布 (用來存影像)
image = Image.new("L", (canvas_size, canvas_size), "white")
draw = ImageDraw.Draw(image)

# 滑鼠畫線
def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill="black")

canvas.bind("<B1-Motion>", paint)

# 清除畫布
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill="white")

# 預測數字
def predict_digit():
    # 轉成 28x28
    img_resized = image.resize((28, 28))
    img_resized = ImageOps.invert(img_resized)  # 反轉: 黑底白字
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1,28,28,1)

    pred = model.predict(img_array)
    digit = np.argmax(pred)
    result_label.config(text=f"辨識結果: {digit}")

# 按鈕
tk.Button(root, text="清除", command=clear_canvas, width=10).grid(row=1, column=0)
tk.Button(root, text="辨識", command=predict_digit, width=10).grid(row=1, column=1)

# 結果顯示
result_label = tk.Label(root, text="辨識結果: ", font=("Helvetica", 16))
result_label.grid(row=1, column=2, columnspan=2)

root.mainloop()
