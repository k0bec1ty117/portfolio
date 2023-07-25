import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
from open_pose import OpenPoseDecoder

# モジュール読み込み 
from openvino.inference_engine import IECore

# プーリング
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):

    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size, strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)

# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)

# get poses from results
def process_results(img, pafs, heatmaps):
    pooled_heatmaps = np.array([[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]])
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # decode poses
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    # output_shape = list(exec_net.output(index=0).partial_shape)
    output_keys = list(exec_net.outputs.keys())#
    output_shape = exec_net.outputs[output_keys[0]].shape
    output_scale = img.shape[1] / output_shape[3], img.shape[0] / output_shape[2]
    # multiply coordinates by scaling factor
    poses[:, :, :2] *= output_scale
    return poses, scores

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
iter = iter(net.outputs)
out_blob_l1 = next(iter)
out_blob_l2 = next(iter)

# インスタンス生成
decoder = OpenPoseDecoder()

# 体部分(19点)
default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
     (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

# 色準備
colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))

# 閾値設定
point_score_threshold = 0.5 

# 入力画像読み込み 
frame = cv2.imread('image/people.jpg')

# 入力データフォーマットへ変換 
image = cv2.resize(frame, (456, 256)) # サイズ変更 
image = image.transpose((2, 0, 1))      # HWC > CHW 
image = np.expand_dims(image, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: image})

# 出力から必要なデータのみ取り出し 
out_l1 = out[out_blob_l1] 
out_l2 = out[out_blob_l2] 

# 出力から姿勢データを取得
poses, scores = process_results(frame, out_l1, out_l2)

# ポーズ描画
img_limbs = np.copy(frame)
for pose in poses:
    points = pose[:, :2].astype(np.int32)
    points_scores = pose[:, 2]

    # 関節部分 描画
    for i, (p, v) in enumerate(zip(points, points_scores)):
        if v > point_score_threshold:
            cv2.circle(frame, (p[0],p[1]), 1, colors[i] , 2)
    # 関節間 線描画
    for i, j in default_skeleton:
        if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
            cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)

# 画像重ね合わせ
cv2.addWeighted(frame, 0.4, img_limbs, 0.6, 0, dst=frame)

# 画像表示 
cv2.imshow('frame', frame)

# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()

#https://docs.openvino.ai/2021.4/notebooks/402-pose-estimation-with-output.html