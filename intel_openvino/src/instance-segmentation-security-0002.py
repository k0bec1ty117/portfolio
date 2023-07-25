import cv2
import numpy as np
 
def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp

def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),(x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask

def mask_rcnn_postprocess(image, out_box, out_mask, img, confidences):
    masks = []
    for box, raw_mask, confidence in zip(out_box, out_mask, confidences):
        box[0] = int(box[0]/img.shape[3] * frame.shape[1])
        box[1] = int(box[1]/img.shape[2] * frame.shape[0])
        box[2] = int(box[2]/img.shape[3] * frame.shape[1])
        box[3] = int(box[3]/img.shape[2] * frame.shape[0])

        if confidence > 0.5: #閾値
            mask = segm_postprocess(box, raw_mask, image.shape[0], image.shape[1])
            masks.append(mask)
    return masks

def overlay_masks(image, masks, ids=None):
    segments_image = image.copy()
    aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    aggregated_colored_mask = np.zeros(image.shape, dtype=np.uint8)
    black = np.zeros(3, dtype=np.uint8)

    for i, mask in enumerate(masks):
        color_idx = i if ids is None else ids[i]
        mask_color = color_palette[color_idx % len(color_palette)].tolist()
        cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
        cv2.bitwise_or(aggregated_colored_mask, np.asarray(mask_color, dtype=np.uint8),dst=aggregated_colored_mask, mask=mask)

    # Fill the area occupied by all instances with a colored instances mask image.
    cv2.bitwise_and(segments_image, black, dst=segments_image, mask=aggregated_mask)
    cv2.bitwise_or(segments_image, aggregated_colored_mask, dst=segments_image, mask=aggregated_mask)
    # Blend original image with the one, where instances are colored.
    # As a result instances masks become transparent.
    cv2.addWeighted(image, 0.5, segments_image, 0.5, 0, dst=image)

    return image

def overlay_class_names(image, boxes, out_labels,confidences):
    labels = [class_labels[i] for i in out_labels]
    white = (255, 255, 255)

    for box, label,confidence in zip(boxes,labels,confidences):
        if confidence > 0.5:
            s = '{}'.format(label.replace("\n", ""))
            textsize = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            position = ((box[:2] + box[2:] - textsize) / 2).astype(int)
            cv2.putText(image, s, tuple(position), cv2.FONT_HERSHEY_SIMPLEX, .5, white, 1)
    return image

# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/instance-segmentation-security-0002/FP32/instance-segmentation-security-0002'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

# ラベル準備
f1 = open('others/coco_80cl.txt', 'r')
class_labels = f1.readlines()

# 色準備
color_palette = np.loadtxt('others/color_palette.csv', delimiter=',')

# 入力画像読み込み 
frame = cv2.imread('image/people.jpg')

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (1024, 768)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し 
out_labels = out["labels"]
out_box = out["boxes"][:,:4]
confidences = out["boxes"][:,4]
out_mask = out["masks"] 

# マスク作成(セグメンテーション色付け)
masks = mask_rcnn_postprocess(frame, out_box, out_mask, img, confidences)

# マスク(色)を重ね合わせ
frame = overlay_masks(frame, masks)

# マスク(ラベル)を重ね合わせ
frame = overlay_class_names(frame, out_box, out_labels, confidences)

# 画像表示 
cv2.imshow('frame', frame)

# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()