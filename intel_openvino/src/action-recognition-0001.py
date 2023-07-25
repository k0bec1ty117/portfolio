import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備 (Encoder)
file_path_en = 'intel/action-recognition-0001/action-recognition-0001-encoder/FP32/action-recognition-0001-encoder'
model_en= file_path_en + '.xml'
weights_en = file_path_en + '.bin'

#モデルの準備 (Decoder)
file_path_de = 'intel/action-recognition-0001/action-recognition-0001-decoder/FP32/action-recognition-0001-decoder'
model_de= file_path_de + '.xml'
weights_de = file_path_de + '.bin'

# モデルの読み込み (Encoder)
net_en = ie.read_network(model=model_en, weights=weights_en)
exec_net_en = ie.load_network(network=net_en, device_name='CPU')

# モデルの読み込み (Decoder)
net_de = ie.read_network(model=model_de, weights=weights_de)
exec_net_de = ie.load_network(network=net_de, device_name='CPU')

# 入出力データのキー取得 (Encoder)
input_blob_en = next(iter(net_en.input_info))
out_blob_en = next(iter(net_en.outputs))

# 入出力データのキー取得 (Decoder)
input_blob_de = next(iter(net_de.input_info))
out_blob_de = next(iter(net_de.outputs))

# カメラ準備 
cap = cv2.VideoCapture("image/action-recognition.mp4")

# ラベル準備
f = open('others/kinetics_400.txt', 'r')
class_labels = f.readlines()

# 準備
encoder_output = []
sample_duration = 16
text = ""

# メインループ 
while True:
    ret, frame = cap.read()

    # 入力データフォーマットへ変換 (Encoder)
    img = cv2.resize(frame, (224, 224)) # サイズ変更 
    img = img.transpose((2, 0, 1))      # HWC > CHW 
    img = np.expand_dims(img, axis=0)   # 次元合せ

    # 推論実行 (Encoder)
    out_en = exec_net_en.infer({input_blob_en: img})

    # 出力から必要なデータのみ取り出し (Encoder)
    out_en = out_en[out_blob_en] 
    encoder_output.append(out_en)

    if len(encoder_output) == sample_duration:

        # 入力データフォーマットへ変換 (Decoder)
        decoder_input = np.concatenate(encoder_output, axis=0)
        decoder_input = decoder_input.transpose((2, 0, 1, 3))
        decoder_input = np.squeeze(decoder_input, axis=3)

        # 推論実行 (Decoder)
        out_de = exec_net_de.infer({input_blob_de: decoder_input})

        # 出力から必要なデータのみ取り出し 
        out_de = out_de[out_blob_de]

        # softmax表現
        exp = np.exp(out_de - np.max(out_de))
        probs = exp / np.sum(exp, axis=None)

        # encorderのリストを空にする
        encoder_output.pop()

        # 出力値が最大のインデックスを得る 
        index_max = np.argmax(probs)
        probs = '{:.2f}'.format(float(np.amax(np.squeeze(probs))) * 100) 
        text = class_labels[index_max].replace("\n", "") + "-" + str(probs) + "%"

    # 文字列描画
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)

   # 何らかのキーが押されたら終了 
    key = cv2.waitKey(1)
    if key != -1:
        break
 
# 終了処理 
cap.release()
cv2.destroyAllWindows()