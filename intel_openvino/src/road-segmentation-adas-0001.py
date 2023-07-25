import cv2
import numpy as np

# モジュール読み込み 
from openvino.inference_engine import IECore

# マスク(色)をつける
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,image[:, :, c] *(1 - alpha) + alpha * color[c] * 255,image[:, :, c])
    return image

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001'
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
img = cv2.resize(frame, (896, 512)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し 
out = out[out_blob] 
out = np.squeeze(out) #サイズ1の次元を全て削除 

# 色準備
color = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]

# 画像準備
frame = cv2.resize(frame, (896, 512))

# マスク(4色)をつける
for i in range(4):
    # 閾値以上の場合、マスクをつける
    mask = out[i] > 0.1
    frame = apply_mask(frame, mask, color[i])

# 画像表示 
cv2.imshow('frame', frame)

# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()