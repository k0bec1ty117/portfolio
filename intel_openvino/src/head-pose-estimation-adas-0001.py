import cv2
import numpy as np
from math import cos, sin
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# 顔方向(yaw row pitch)を描画
def draw_axis(img, yaw, pitch, roll, xcenter, ycenter, size = 100):

    # yaw row pitch取得
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180
    roll = roll * np.pi / 180

    # X-Axis (赤)
    x1 = size * (cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll)) + xcenter
    y1 = size * (cos(pitch) * sin(roll)) + ycenter

    # Y-Axis (緑)
    x2 = size * (cos(roll) * sin(yaw) * sin(pitch) + cos(yaw) * sin(roll)) + xcenter
    y2 = ycenter - size * (cos(pitch) * cos(roll))

    # Z-Axis(青)
    x3 = size * (sin(yaw) * cos(pitch)) + xcenter
    y3 = size * (sin(pitch)) + ycenter

    # 線描画
    cv2.line(img, (int(xcenter), int(ycenter)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(xcenter), int(ycenter)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(xcenter), int(ycenter)), (int(x3),int(y3)),(255,0,0),2)

    return img

# IEコアの初期化
ie = IECore()

#モデルの準備 (顔検出)
file_path_face = 'intel/face-detection-retail-0004/FP32/face-detection-retail-0004'
model_face= file_path_face + '.xml'
weights_face = file_path_face + '.bin'

#モデルの準備 (顔方向)
file_path_head = 'intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'
model_head= file_path_head + '.xml'
weights_head = file_path_head + '.bin'

# モデルの読み込み (顔検出)
net_face = ie.read_network(model=model_face, weights=weights_face)
exec_net_face = ie.load_network(network=net_face, device_name='CPU')

# モデルの読み込み (顔方向)
net_head = ie.read_network(model=model_head, weights=weights_head)
exec_net_head = ie.load_network(network=net_head, device_name='CPU')

# 入出力データのキー取得 (顔検出)
input_blob_face = next(iter(net_face.input_info))
out_blob_face = next(iter(net_face.outputs))

# 入出力データのキー取得 (顔方向)
input_blob_head = next(iter(net_head.input_info))
out_blob_head = next(iter(net_head.outputs))

# モデルの読み込み (顔検出)
net_face = ie.read_network(model=model_face, weights=weights_face)
exec_net_face = ie.load_network(network=net_face, device_name='CPU')

# モデルの読み込み (顔方向)
net_head = ie.read_network(model=model_head, weights=weights_head)
exec_net_head = ie.load_network(network=net_head, device_name='CPU')

# 入出力データのキー取得 (顔検出)
input_blob_face = next(iter(net_face.input_info))
out_blob_face = next(iter(net_face.outputs))

# 入出力データのキー取得 (顔方向)
input_blob_head = next(iter(net_head.input_info))
iter_head = iter(net_head.outputs)
out_blob_p = next(iter_head)
out_blob_r = next(iter_head)
out_blob_y = next(iter_head)

# 入力画像読み込み 
frame = cv2.imread('image/people.jpg')

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (300, 300)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net_face.infer({input_blob_face: img})

# 出力から必要なデータのみ取り出し 
out = out[out_blob_face]
out = np.squeeze(out) #サイズ1の次元を全て削除 

# 検出されたすべての顔領域に対して１つずつ処理 
for detection in out:
    # conf値の取得 
    confidence = float(detection[2])

    # バウンディングボックス座標を入力画像のスケールに変換 
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])

    # conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示 
    if confidence > 0.5:
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > frame.shape[1]:
            xmax = frame.shape[1]
        if ymax > frame.shape[0]:
            ymax = frame.shape[0]
        
        # バウンディングボックス表示 
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)

        # 顔領域のみ切り出し 
        frame_face = frame[ymin:ymax, xmin:xmax] 

        # 入力データフォーマットへ変換 
        img = cv2.resize(frame_face, (60, 60)) # サイズ変更 
        img = img.transpose((2, 0, 1))      # HWC > CHW 
        img = np.expand_dims(img, axis=0)   # 次元合せ

        # 推論実行 
        out = exec_net_head.infer({input_blob_head: img})

        #出力から必要なデータのみ取り出し 
        out_p = out[out_blob_p] 
        pitch = np.squeeze(out_p) #サイズ1の次元を全て削除 

        out_r = out[out_blob_r] 
        roll = np.squeeze(out_r) #サイズ1の次元を全て削除 

        out_y = out[out_blob_y] 
        yaw = np.squeeze(out_y) #サイズ1の次元を全て削除 

        # 顔の中心位置
        head_center = [(xmax + xmin)//2, (ymax + ymin)//2]

        # yaw row pitch を描画
        frame = draw_axis(frame, yaw, pitch, roll, head_center[0], head_center[1], 100)

# 画像表示 
cv2.imshow('frame', frame)

# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()