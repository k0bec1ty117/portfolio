import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/license-plate-recognition-barrier-0001/FP32/license-plate-recognition-barrier-0001'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
iter_input = iter(net.input_info)
input_blob_data = next(iter_input)
input_blob_ind = next(iter_input)
out_blob = next(iter(net.outputs))

# 入力画像読み込み 
frame = cv2.imread('image/license_plate.jpg')

# 入力データ準備
seq_ind = np.array([[1]]*88)

# ラベルデータ準備
f = open('others/license_plate.txt', 'r')
labels = f.readlines()

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (94, 24)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out_ind = exec_net.infer({input_blob_ind:seq_ind })
out = exec_net.infer({input_blob_data:img})

# 出力から必要なデータのみ取り出し 
out = out[out_blob] 
out = np.squeeze(out) #サイズ1の次元を全て削除 

# 準備
text = []

# 検出されたすべての文字に対して１つずつ処理 
for detection in out:
    if int(detection) != -1:
        label = labels[int(detection)]
        label = label.replace("\n", "")
        text.append(label)

# 表示
print("".join(text)) 
