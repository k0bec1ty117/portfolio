import cv2
import numpy as np
from codec import CTCCodec

# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/handwritten-japanese-recognition-0001/FP32/handwritten-japanese-recognition-0001'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

#ラベル準備
with open("others/kondate_nakayosi.txt", 'r', encoding='utf-8') as f:
    character = [line.strip('\n') for line in f]

# インスタンス生成
codec = CTCCodec(character, None, 20)

# 入力画像読み込み 
frame = cv2.imread('image/handwritten.jpg', cv2.IMREAD_GRAYSCALE)
ratio = float(frame.shape[1]) / float(frame.shape[0])
height = 96
width = 2000
tw = int(height * ratio)

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (tw, height)) # サイズ変更
img = np.expand_dims(img, axis=0)   # 次元合せ
img = np.pad(img, ((0, 0), (0, height - img.shape[1]), (0, width - img.shape[2])), mode='edge') #パディング
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し 
out = out[out_blob] 

# 出力から CTCデコード
result = codec.decode(out)

# 表示
print(*result)
