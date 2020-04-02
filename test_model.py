from flask import Flask
from flask_cors import CORS
from flask import request
import json

#svm有关
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


app = Flask(__name__)
#允许所有跨域请求
CORS(app, resources=r'/*')

#模型
clf = None

'''
C:错误项的惩罚系数，越大对误判样本惩罚越大，训练集上准确率越高，但是泛化能力减弱；
减小c泛化能力加强，但在训练集上的准确率降低

kernal:采用核函数的类型可选的有：linear--线性核函数 ;poly--多项式核函数；rbf--径向核函数/高斯函数；sigmod:sigmod核函数；precomputed:矩阵函数

max_iter: 最大迭代系数，-1表示不限制
'''
def getSVM(C=1, kernel="rbf", max_iter=-1):
    return svm.SVC(C=C, kernel=kernel, max_iter=max_iter, gamma='auto')


#这里一定要写methods请求方式，否则会发生405错误
@app.route('/SVM_Train', methods=['post'])
def svm_train():
    raw = request.get_data(as_text=True)
    if len(raw) == 0:
        return json.dumps({'status': 'fail', 'msg': '服务器没有接收到任何Json数据，请确保传输的数据格式是json'}, ensure_ascii=False)
    datas = json.loads(raw)
    #获取训练数据
    str_x = datas['train_x']
    str_y = datas['train_y']
    test_size = float(datas['test_size'])

    kernel = 'rbf'
    c = 0.001
    max_iter = -1
    #获取参数
    if 'kernel' in datas:
        if datas['kernel'] != '':
            kernel = datas['kernel'].strip()
    if 'c' in datas:
        if datas['c'] !='':
            c = float(datas['c'].strip())
    if 'max_iter' in datas:
        if datas['max_iter'] != '':
            max_iter = float(datas['max_iter'].strip)

    if type(str_x) == str:
        str_x = json.loads(str_x)
    if type(str_y) == str:
        str_y = json.loads(str_y)

    x = [[float(n) for n in val] for val in str_x]
    y = [int(n) for n in str_y]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)

    global clf
    if clf is None:
        clf=getSVM(C=c, kernel=kernel, max_iter=max_iter)

    clf.fit(train_x, train_y)
    prediction = clf.predict(test_x)
    acc = metrics.accuracy_score(prediction, test_y)

    return json.dumps({'status': 'done', 'accuracy': acc})


@app.route('/SVM_Predict', methods=['post'])
def predict_svm():
    global clf
    if clf is None:
        return json.dumps({'status': 'fail', 'msg': '在预测之前必须先训练模型'}, ensure_ascii=False)
    req = request
    raw = request.get_data(as_text=True)
    if raw is None:
        return json.dumps({'status': 'fail', 'msg': '服务器没有接收到任何Json数据，请确保传输的数据格式是json'})
    datas = json.loads(raw)

    #这里判断是否已经将字符串转换成对应的数组，没有继续转换
    if type(datas['predict_x']) == str:
        xx = json.loads(datas['predict_x'])
        x = [[float(i) for i in j] for j in xx]

    else:
        x = [[float(i) for i in j] for j in datas['predict_x']]

    #注意不能直接返回numpy,因为这不可json化
    res = clf.predict(x)
    return json.dumps({'status': 'done', 'result': res.tolist()})


def irisTest():
    iris = datasets.load_iris()
    x = iris.data
    targets = iris.target
    train_x, test_x, train_y, test_y = train_test_split(x, targets, test_size=0.2)
    print('train_datas:')
    #for j in train_x:
    #print([','.join(str(i) for i in j) for j in train_x])
    for j in x:
        print('[', end='')
        print(','.join(str(k) for k in j), end='')
        print('],')
    print('[' + ','.join(str(k) for k in targets)+']')
    print('test_datas:')
    print()
    #print([','.join(str(i) for i in j) for j in test_x])
    for j in test_x:
        print('[', end='')
        print(','.join(str(k) for k in j), end='')
        print('],')
    print('[' + ','.join(str(k) for k in test_y)+']')

'''
    global clf
    if clf is None:
        clf=getSVM(C=1)
    clf.fit(train_x, train_y)
    prediction = clf.predict(test_x)
    print('the accuracy on iris datasets is: {0}'.format(metrics.accuracy_score(prediction, test_y)))
    #test_x = [i[0] for i in x]
    #test_y = [i[1] for i in x]
    #plt.plot(test_x, test_y, 'ob')
    #plt.show()
'''


@app.route('/test', methods=['post', 'get', 'put'])
def testLink():
    print('successsfully linked')
    return json.dumps({'status':'done'})


from views.svm_view import svm_api

if __name__ == '__main__':
    #irisTest()
    port=input('please input port:')
    CORS(svm_api, resources=r'/*')
    app.register_blueprint(svm_api, url_prefix='/svm')
    app.run('0.0.0.0',port=port)
