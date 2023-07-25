import cv2
import numpy as np
from codec import CTCCodec

# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/text-recognition-0012/FP32/text-recognition-0012'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

# ラベル準備
alphanumeric = "0123456789abcdefghijklmnopqrstuvwxyz#"

# インスタンス生成
decoder = CTCCodec(alphanumeric, None, 10) 

# 入力画像読み込み 
frame = cv2.imread('image/text_re1.jpg')

# 入力データフォーマットへ変換 
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #グレースケール
img = cv2.resize(img, (120, 32)) # サイズ変更 
img = np.expand_dims(img, axis=0)   # 次元合せ
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し 
out = out[out_blob]

# 出力から CTCデコード
result = decoder.ctc_decode(out)

# 表示
print(*result)