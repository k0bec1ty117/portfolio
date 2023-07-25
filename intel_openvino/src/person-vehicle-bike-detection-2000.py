import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/person-vehicle-bike-detection-2000/FP32/person-vehicle-bike-detection-2000'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

# 入力画像読み込み 
frame = cv2.imread('image/car.jpg')

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (256, 256)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し 
out = out[out_blob] 
out = np.squeeze(out) #サイズ1の次元を全て削除 

# 検出されたすべての顔領域に対して１つずつ処理 
for detection in out:
    # label値の取得 
    label = int(detection[1])

    # conf値の取得 
    confidence = float(detection[2])

    # バウンディングボックスの色を指定
    colors = {0:(50,205,50), 1:(220,20,60), 2:(35,59,108)}

    # バウンディングボックス座標を入力画像のスケールに変換 
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])

    # conf値より大きい場合バウンディングボックス表示 
    if confidence > 0.1:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=colors[label], thickness=3)
 
    # 画像表示 
    cv2.imshow('frame', frame)
 
# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()