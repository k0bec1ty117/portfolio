import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

# 入力画像読み込み 
frame = cv2.imread('image/photo_face.jpg')

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (60, 60)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し 
out = out[out_blob] 
out = np.squeeze(out) #サイズ1の次元を全て削除 

out = out.reshape(35,2)

# 検出されたすべての顔領域に対して１つずつ処理 
for detection in out:
    x = int(detection[0]*frame.shape[1])
    y = int(detection[1]*frame.shape[0])
    cv2.circle(frame, (x, y), 5, (0, 0, 255), thickness=-1)

    # 画像表示 
    cv2.imshow('frame', frame)
 
# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()