import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/person-attributes-recognition-crossroad-0230/FP32/person-attributes-recognition-crossroad-0230'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
iter_output = iter(net.outputs)
out_blob_label = next(iter_output)
out_blob_top = next(iter_output)
out_blob_bottom = next(iter_output)

# 入力画像読み込み 
frame = cv2.imread('image/standing_person.jpg')

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (80, 160)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し (ラベル)
out_label = out[out_blob_label] 
out_label = np.squeeze(out_label) 

# 出力から必要なデータのみ取り出し (上半身)
out_top = out[out_blob_top] 
out_top = np.squeeze(out_top) 

# 出力から必要なデータのみ取り出し (下半身)
out_bottom = out[out_blob_bottom] 
out_bottom = np.squeeze(out_bottom) 

# ラベル準備
labels = ["is_male", "has_bag", "has_backpack", "has_hat", "has_longsleeves", "has_longpants", "has_longhair", "has_coat_jacket"]

# 検出されたすべての顔領域に対して１つずつ処理 
for index, detection in enumerate(out_label):
    # conf値の取得 
    confidence = float(detection)

    # conf値より大きい場合ラベル表示 
    if confidence > 0.5:
        print(labels[index])

#上半身の座標
x1, y1 = int(out_top[0]*frame.shape[1]), int(out_top[1]*frame.shape[0])
cv2.circle(frame, (x1, y1), 5, (0, 0, 255), thickness=-1)

#下半身の座標
x2, y2 = int(out_bottom[0]*frame.shape[1]), int(out_bottom[1]*frame.shape[0])
cv2.circle(frame, (x2, y2), 5, (0, 0, 255), thickness=-1)

# 画像表示 
cv2.imshow('frame', frame)
 
# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()
