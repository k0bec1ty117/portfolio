# micro:bit_x_ml

## 概要

人がジェスチャーを行うとリアルタイムに推論するシステム

`実稼働画面をキャプチャー`

### 学習時

![image](https://user-images.githubusercontent.com/89716609/219562041-d5650eab-eff7-4d0d-8ba3-6be6571088bd.png)

- micro:bitの3軸加速度センサーを利用し`data`を取得
- 加速度センサ x, y, zデータをローカルで立ち上げたフロントエンドから、URLクエリパラメータを用いてバックエンドに`dataset`を送信
- 加速度センサ x, y, zデータと正解ラベルから50個ごとの平均と標準偏差を計算し、各ジェスチャーの特徴を抽出
- 前処理(`svm_learning.ipynb`)を行ったデータを訓練データ・テストデータに分け、Support Vector Machine(SVM)で`model`を作成

<br>

### 推論時

![image](https://user-images.githubusercontent.com/89716609/219563745-78d0f230-cca2-48c8-849d-821eb7e57a52.png)

- 加速度センサ x, y, zデータから、50個ごとの平均と標準偏差を計算した`前処理後data`をバックエンドに送信
- 学習済み`model`を用いて、バックエンドで`predict`を行い、推論結果をフロントエンドに送信
- リアルタイムで取得したデータの推論結果を、フロントエンドに表示

<br>

## 対応OS
- Windows10（version1706以降）
- Mac (OS X Yosemite以降)

## 対応ブラウザ
- Chromeのみ

## 使用した技術
- 加速度センサー
- micro:bit
- HTML
- JavaScript
- BLE
- WebBluetooth
- Python
- Pandas
- scikit-learn
- SVM
- Flask
- GET/POST

