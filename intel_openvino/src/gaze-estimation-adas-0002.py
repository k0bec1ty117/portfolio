import cv2
import numpy as np
from math import cos, sin, radians
 
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

# 目の方向を描画
def draw_axis_eye(img, out, xcenter, ycenter, size):

    # yaw row pitch取得
    yaw, pitch, roll = out[0], out[1], out[2]

    vcos = cos(radians(roll))
    vsin = sin(radians(roll))
    x = xcenter + size * (yaw * vcos + pitch * vsin)
    y = ycenter - size * (yaw * vsin + pitch * vcos)

    # 線描画
    cv2.line(img, (int(xcenter), int(ycenter)), (int(x),int(y)), (0,255,255), 2) #黄

    return img

# IEコアの初期化
ie = IECore()

#モデルの準備(顔検出)
file_path_face = 'intel/face-detection-retail-0004/FP32/face-detection-retail-0004'
model_face= file_path_face + '.xml'
weights_face = file_path_face + '.bin'

#モデルの準備(ランドマーク)
file_path_land = 'intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002'
model_land = file_path_land + '.xml'
weights_land = file_path_land + '.bin'

#モデルの準備(顔方向)
file_path_head = 'intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'
model_head= file_path_head + '.xml'
weights_head = file_path_head + '.bin'

#モデルの準備(目線)
file_path_gaze = 'intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'
model_gaze= file_path_gaze + '.xml'
weights_gaze = file_path_gaze + '.bin'

# モデルの読み込み(顔検出)
net_face = ie.read_network(model=model_face, weights=weights_face)
exec_net_face = ie.load_network(network=net_face, device_name='CPU')

# モデルの読み込み(ランドマーク)
net_land = ie.read_network(model=model_land, weights=weights_land)
exec_net_land = ie.load_network(network=net_land, device_name='CPU')

# モデルの読み込み(顔方向)
net_head = ie.read_network(model=model_head, weights=weights_head)
exec_net_head = ie.load_network(network=net_head, device_name='CPU')

# モデルの読み込み(目線)
net_gaze = ie.read_network(model=model_gaze, weights=weights_gaze)
exec_net_gaze = ie.load_network(network=net_gaze, device_name='CPU')

# 入出力データのキー取得(顔検出)
input_blob_face = next(iter(net_face.input_info))
out_blob_face = next(iter(net_face.outputs))

# 入出力データのキー取得 (ランドマーク)
input_blob_land = next(iter(net_land.input_info))
out_blob_land = next(iter(net_land.outputs))

# 入出力データのキー取得 (顔方向)
input_blob_head = next(iter(net_head.input_info))
iter_head = iter(net_head.outputs)
out_blob_p = next(iter_head)
out_blob_r = next(iter_head)
out_blob_y = next(iter_head)

# 入出力データのキー取得 (目線)
iter_gaze = iter(net_gaze.input_info)
input_blob_angles = next(iter_gaze)
input_blob_left = next(iter_gaze)
input_blob_right = next(iter_gaze)

# 入力画像読み込み 
frame = cv2.imread('image/people.jpg')

# 入力データフォーマットへ変換 (顔検出)
img = cv2.resize(frame, (300, 300)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 (顔検出)
out = exec_net_face.infer({input_blob_face: img})

# 出力から必要なデータのみ取り出し (顔検出)
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
        if xmin < 0:xmin = 0
        if ymin < 0:ymin = 0
        if xmax > frame.shape[1]:xmax = frame.shape[1]
        if ymax > frame.shape[0]:ymax = frame.shape[0]
        
        # 顔領域のみ切り出し 
        frame_face = frame[ymin:ymax, xmin:xmax] 

        # 顔の中心位置
        head_center = [(xmin+xmax)//2, (ymin+ymax)//2]

        # 入力データフォーマットへ変換 (ランドマーク)
        img = cv2.resize(frame_face, (60, 60)) # サイズ変更 
        img = img.transpose((2, 0, 1))      # HWC > CHW 
        img = np.expand_dims(img, axis=0)   # 次元合せ

        # 推論実行 (ランドマーク)
        out = exec_net_land.infer({input_blob_land: img})

        # 出力から必要なデータのみ取り出し (ランドマーク)
        out = out[out_blob_land] 
        out = np.squeeze(out) #サイズ1の次元を全て削除 
        out = out.reshape(35,2)

        # 左目の座標
        xmax_left = int(out[0][0] * frame_face.shape[1])
        ymax_left = int(out[0][1] * frame_face.shape[0])
        xmin_left = int(out[1][0] * frame_face.shape[1])
        ymin_left = int(out[1][1] * frame_face.shape[0])

        if xmin_left < 0:xmin_left = 0
        if ymin_left < 0:ymin_left = 0
        if xmax_left > frame_face.shape[1]:xmax_left = frame_face.shape[1]
        if ymax_left > frame_face.shape[0]:ymax_left = frame_face.shape[0]

        # 右目の座標
        xmax_right = int(out[2][0] * frame_face.shape[1])
        ymax_right = int(out[2][1] * frame_face.shape[0])
        xmin_right = int(out[3][0] * frame_face.shape[1])
        ymin_right = int(out[3][1] * frame_face.shape[0])

        if xmin_right < 0:xmin_right = 0
        if ymin_right < 0:ymin_right = 0
        if xmax_right > frame_face.shape[1]:xmax_right = frame_face.shape[1]
        if ymax_right > frame_face.shape[0]:ymax_right = frame_face.shape[0]

        # 左・右目の中心座標
        eye_center_left = [(xmax_left + xmin_left) // 2, (ymax_left + ymin_left) // 2]
        eye_center_right = [(xmax_right + xmin_right) // 2, (ymax_right + ymin_right) // 2]

        # 目の大きさ
        eye_size = xmax_left - xmin_left

        # 左・右目を抽出
        frame_left_eye = frame_face[eye_center_left[1] - eye_size: eye_center_left[1] + eye_size, eye_center_left[0] - eye_size: eye_center_left[0] + eye_size]
        frame_right_eye = frame_face[eye_center_right[1] - eye_size: eye_center_right[1] + eye_size, eye_center_right[0] - eye_size: eye_center_right[0] + eye_size]

        # 入力データフォーマットへ変換 (顔方向)
        img = cv2.resize(frame_face, (60, 60)) # サイズ変更 
        img = img.transpose((2, 0, 1))      # HWC > CHW 
        img = np.expand_dims(img, axis=0)   # 次元合せ

        # 推論実行 (顔方向)
        out = exec_net_head.infer({input_blob_head: img})

        #出力から必要なデータのみ取り出し (顔方向)
        out_p = out[out_blob_p] 
        pitch = np.squeeze(out_p) #サイズ1の次元を全て削除 

        out_r = out[out_blob_r] 
        roll = np.squeeze(out_r) #サイズ1の次元を全て削除 

        out_y = out[out_blob_y] 
        yaw = np.squeeze(out_y) #サイズ1の次元を全て削除 

        # 顔の方向を描画
        frame = draw_axis(frame, yaw, pitch, roll, head_center[0], head_center[1], 100)

        #顔方向のyaw pitch row 
        euler = np.array([yaw, pitch, roll])

        img_left = cv2.resize(frame_left_eye, (60, 60)) # サイズ変更 
        img_left = img_left.transpose((2, 0, 1))      # HWC > CHW 
        img_left = np.expand_dims(img_left, axis=0)   # 次元合せ

        img_right = cv2.resize(frame_right_eye, (60, 60)) # サイズ変更 
        img_right = img_right.transpose((2, 0, 1))      # HWC > CHW 
        img_right = np.expand_dims(img_right, axis=0)   # 次元合せ

        # 推論実行 
        out = exec_net_gaze.infer({input_blob_left: img_left, input_blob_right: img_right, input_blob_angles: euler})
        # 出力から必要なデータのみ取り出し 
        out = out["gaze_vector"] 
        out = np.squeeze(out) #サイズ1の次元を全て削除 

        # 出力ベクトルは正規化
        out = out / np.linalg.norm(out) 

        # バウンディングボックス表示(目)
        cv2.rectangle(frame_face, (eye_center_left[0] - eye_size, eye_center_left[1] - eye_size), (eye_center_left[0] + eye_size, eye_center_left[1] + eye_size), (255, 255, 255), 1)
        cv2.rectangle(frame_face, (eye_center_right[0] - eye_size, eye_center_right[1] - eye_size), (eye_center_right[0] + eye_size, eye_center_right[1] + eye_size), (255, 255, 255), 1)

        # 左目の方向
        frame = draw_axis_eye(frame, out, eye_center_left[0] + xmin, eye_center_left[1] + ymin, 80)
        # 右目の方向
        frame = draw_axis_eye(frame, out, eye_center_right[0] + xmin, eye_center_right[1] + ymin, 80)

# 画像表示 
cv2.imshow('frame', frame)
 
# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()

