import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/asl-recognition-0004/FP32/asl-recognition-0004'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

# カメラ準備 
cap = cv2.VideoCapture("image/sign-language.mp4")

#ラベル準備
f = open('others/ms_asl_100.txt', 'r')
class_labels = f.readlines()

images = []
sample_duration = 16
text = ""

# メインループ 
while True:
    ret, frame = cap.read()

    img = cv2.resize(frame, (224, 224)) # サイズ変更 
    img = img.transpose((2, 0, 1))      # HWC > CHW 
    img = np.expand_dims(img, axis=0)   # 次元合せ
    images.append(img)

    if len(images) == sample_duration:

        # 入力データフォーマットへ変換
        input = np.concatenate(images, axis=0)
        input = input.transpose((1, 0, 2, 3)) # TCHW > CTHW
        input = np.expand_dims(input, axis=0)
        images.pop(0)

        # 推論実行 
        out = exec_net.infer({input_blob: input})

        # 出力から必要なデータのみ取り出し
        out = out[out_blob] 
        out = np.squeeze(out, axis=0)

        # softmax表現
        exp = np.exp(out - np.max(out))
        probs = exp / np.sum(exp, axis=None)

        # 出力値が最大のインデックスを得る 
        index_max = np.argmax(probs)
        text = class_labels[index_max].replace("\n", "").replace('"', "").replace(',', "")

    # 文字列描画
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 画像表示
    cv2.imshow('frame', frame)

   # 何らかのキーが押されたら終了 
    key = cv2.waitKey(1)
    if key != -1:
        break
 
# 終了処理 
cap.release()
cv2.destroyAllWindows()