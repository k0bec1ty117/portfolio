import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))

# 入力画像読み込み 
frame = cv2.imread('image/text.jpg')

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (704, 704)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し 
out = out["boxes"] 
out = np.squeeze(out) #サイズ1の次元を全て削除 

#検出されたすべての顔領域に対して１つずつ処理 
for detection in out:
    # conf値の取得 
    confidence = float(detection[4])

    # バウンディングボックス座標を入力画像のスケールに変換 
    xmin = int(detection[0]/img.shape[3] * frame.shape[1])
    ymin = int(detection[1]/img.shape[2] * frame.shape[0])
    xmax = int(detection[2]/img.shape[3] * frame.shape[1])
    ymax = int(detection[3]/img.shape[2] * frame.shape[0])

    # conf値より大きい場合バウンディングボックス表示 
    if confidence > 0.3:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)
 
    # 画像表示 
    cv2.imshow('frame', frame)
 
# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()