from flask import Flask, jsonify, request
import pickle
import os, datetime
from sklearn.svm import LinearSVC
import csv

#モデル読み込み
filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))

#ファイル情報
time_stamp = os.stat(filename).st_mtime
time_stamp = datetime.datetime.fromtimestamp(time_stamp).strftime(' %Y-%m-%d %H:%M:%S')
file_info = filename  + time_stamp

#Flaskのインスタンス生成
app = Flask(__name__)

#推論時のルーティング
@app.route('/predict', methods=['GET']) 
def predict():
    #推論用データ取得
    mean_x = request.args.get('mean_x')
    mean_y = request.args.get('mean_y')
    mean_z = request.args.get('mean_z')
    std_x = request.args.get('std_x')
    std_y = request.args.get('std_y')
    std_z = request.args.get('std_z')    
    X_test = [[float(mean_x), float(mean_y), float(mean_z), float(std_x), float(std_y), float(std_z)]]

    #推論
    y_pred = model.predict(X_test)
    output = y_pred[0]

    #推論結果を返す
    return jsonify({'out': output, 'info': file_info})

#学習データ作成時のルーティング
@app.route('/training', methods=['POST'])
def write_csv():
    #学習用データ取得
    microbit_data = request.form['data']

    #csvファイル保存
    with open('microbit_data.csv', 'a') as f:
        print(microbit_data, file=f)

    #文字列を返す
    return jsonify({'info': 'Completed saving the csv file'})

#CORSエラー対策
@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*') 
  return response

#Flask起動
app.run(debug=True, host='0.0.0.0', port='5001')
