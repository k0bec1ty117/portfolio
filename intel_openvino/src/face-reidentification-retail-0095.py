import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# 顔画像の整列   顔が正面を向いて整列されている場合に最良の結果を出す
def align_face(face_frame, landmarks):
    left_eye_x, left_eye_y, right_eye_x, right_eye_y = landmarks[:4].tolist()
    # 目中心間の角度を計算
    dy = right_eye_y - left_eye_y
    dx = right_eye_x - left_eye_x
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # 顔画像の中心
    center = (face_frame.shape[0] // 2, face_frame.shape[1] // 2)
    h, w, _ = face_frame.shape

    # 回転・拡大縮小の為に行列を取得
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_frame, M, (w, h))

    return aligned_face

def cos_similarity(X, Y):
    Y = Y.T
    # (1, 256) x (256, n) = (1, n)
    return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y, axis=0))

# IEコアの初期化
ie = IECore()

#モデルの準備(顔検出)
file_path_face = 'intel/face-detection-retail-0004/FP32/face-detection-retail-0004'
model_face= file_path_face + '.xml'
weights_face = file_path_face + '.bin'

#モデルの準備(ランドマーク)
file_path_land = 'intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
model_land= file_path_land + '.xml'
weights_land = file_path_land + '.bin'

#モデルの準備(顔再識別)
file_path = 'intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095'
model_reid = file_path + '.xml'
weights_reid = file_path + '.bin'

# モデルの読み込み(顔検出)
net_face = ie.read_network(model=model_face, weights=weights_face)
exec_net_face = ie.load_network(network=net_face, device_name='CPU')

# モデルの読み込み(ランドマーク)
net_land = ie.read_network(model=model_land, weights=weights_land)
exec_net_land = ie.load_network(network=net_land, device_name='CPU')

# モデルの読み込み(顔再識別)
net_reid = ie.read_network(model=model_reid, weights=weights_reid)
exec_net_reid = ie.load_network(network=net_reid, device_name='CPU')

# 入出力データのキー取得 (顔検出)
input_blob_face = next(iter(net_face.input_info))
out_blob_face = next(iter(net_face.outputs))

# 入出力データのキー取得 (ランドマーク)
input_blob_land = next(iter(net_land.input_info))
out_blob_land = next(iter(net_land.outputs))

# 入出力データのキー取得 (顔再識別)
input_blob_reid = next(iter(net_reid.input_info))
out_blob_reid = next(iter(net_reid.outputs))

# 入力画像読み込み 
frame = cv2.imread('image/people.jpg')

# 入力データフォーマットへ変換 
img_face = cv2.resize(frame, (300, 300)) # サイズ変更 
img_face = img_face.transpose((2, 0, 1))      # HWC > CHW 
img_face = np.expand_dims(img_face, axis=0)   # 次元合せ

# 推論実行 
out_face = exec_net_face.infer({input_blob_face: img_face})

# 出力から必要なデータのみ取り出し 
out_face = out_face[out_blob_face] 
out_face = np.squeeze(out_face) #サイズ1の次元を全て削除 

# display aligned faces
feature_vecs = []
aligned_faces = []

# 検出されたすべての顔領域に対して１つずつ処理 
for detection in out_face:
    # conf値の取得 
    confidence = float(detection[2])

    # バウンディングボックス座標を入力画像のスケールに変換 
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])

    if confidence > 0.5:
        # 顔検出領域は入力画像範囲内に補正する。特にminは補正しないとエラーになる 
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > frame.shape[1]:
            xmax = frame.shape[1]
        if ymax > frame.shape[0]:
            ymax = frame.shape[0]

        # 顔領域のみ切り出し 
        face_img = frame[ymin:ymax, xmin:xmax]

        # 入力データフォーマットへ変換 
        frame_face = cv2.resize(face_img, (48, 48)) # サイズ変更 
        frame_face = frame_face.transpose((2, 0, 1))      # HWC > CHW 
        frame_face = np.expand_dims(frame_face, axis=0)   # 次元合せ

        # 推論実行 
        out_land = exec_net_land.infer({input_blob_land: frame_face})

        # 出力から必要なデータのみ取り出し 
        out_land = out_land[out_blob_land] 
        out_land = np.squeeze(out_land) #サイズ1の次元を全て削除 

        aligned_face = face_img.copy()
        aligned_face = align_face(aligned_face, out_land)
        aligned_faces.append(aligned_face)

        # 入力データフォーマットへ変換 (顔再識別)
        aligned_face = cv2.resize(aligned_face, (128, 128)) # サイズ変更 
        aligned_face = aligned_face.transpose((2, 0, 1))      # HWC > CHW 
        aligned_face = np.expand_dims(aligned_face, axis=0)   # 次元合せ

        # 推論実行 (顔再識別)
        out_reid = exec_net_reid.infer({input_blob_reid: aligned_face})

        # 出力から必要なデータのみ取り出し 
        out_reid = out_reid[out_blob_reid] 
        out_reid = np.squeeze(out_reid) #サイズ1の次元を全て削除 
        feature_vecs.append(out_reid)

#ターゲット指定
target_vec = feature_vecs[0]

# 類似度を計算
similarity = cos_similarity(target_vec, np.array(feature_vecs))
print("similarity: {}".format(similarity))

# 類似度が最も高いものを表示
for face_id, aligned_face in enumerate(aligned_faces):
    face_tmp = aligned_face.copy()
    if face_id == similarity.argmax():
        # ターゲット画像表示 
        cv2.imshow('target_face', face_tmp)

# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()