import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
iter = iter(net.outputs)
out_blob_age = next(iter)
out_blob_gender = next(iter)

# 入力画像読み込み 
frame = cv2.imread('image/face.jpg')

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (62, 62)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し (age)
out_age = out[out_blob_age] 
out_age = np.squeeze(out_age) #サイズ1の次元を全て削除 

# 出力から必要なデータのみ取り出し (gender)
out_gender = out[out_blob_gender] 
out_gender = np.squeeze(out_gender) #サイズ1の次元を全て削除 

# 出力値が最大のインデックスを得る 
index_max_gender = np.argmax(out_gender)

# 文字列描画
cv2.putText(frame, ["female", "male"][index_max_gender], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(frame, str(int(out_age*100)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# 画像表示 
cv2.imshow('image', frame)

# 終了処理 
cv2.waitKey(0)
cv2.destroyAllWindows()
