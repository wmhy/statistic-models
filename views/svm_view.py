

from flask import Blueprint
from flask_cors import CORS
from flask import request
import json
from views import svm_api

from utils import get_json_data, type_format, check_aquired_args
from models.SVM import get_svm, get_trained_svm, train, predict

@svm_api.route('/svm_test', methods=['post'])
def svm_test():
    '''
    svm模块的测试接口
    Returns:

    '''
    return json.dumps({'status': 'done', 'msg': '你好'}, ensure_ascii=False)

@svm_api.route('/train', methods=['post'])
def svm_train():
    '''

    Returns:
    acc: 在测试集上的精度
    '''
    #参数及默认值
    args = {
        'train_x': None, 'train_y': None,
        'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'auto', 'coef0': 0.0,
        'shrinking': True, 'probability': False, 'tol': 0.001, 'cache_size': 200,
        'class_weight': None, 'verbose': False, 'max_iter': -1,
        'decision_function_shape': 'ovr', 'random_state': None,
        'test_size':0.1
    }
    #必须参数, 不要在arg中给必须参数赋默认值
    acq = ['train_x', 'train_y']
    #参数限制暂时不写
    lim = []

    #参数类型，注意，这里忽略了同时可以为str或者float的参数， 例如gamma
    #对于list的参数，第一个元素表示本身，第二个表示第一维索引的元素类型，以此类推
    types ={
        'train_x': [list, list, float], 'train_y':[list, float],
        'C': float, 'kernel': str, 'degree': int,  'coef0': float,
        'shrinking': bool, 'probability': bool, 'tol': float, 'cache_size': int,
        'class_weight': str, 'verbose': bool, 'max_iter': int, 'decision_function_shape': str, 'random_state': int,
        'test_size': float
    }

    args = get_json_data(raw=request.get_data(as_text=True), args=args)
    if args is None:
        return json.dumps({'state': 'fail', 'msg': '未接收到任何json数据'}, ensure_ascii=False)
    missed_arg = check_aquired_args(args, acq)
    if missed_arg is not None:
        return json.dumps({'state': 'fail', 'msg': '缺少必须参数'}, ensure_ascii=False)
    args = type_format(args=args, types=types)

    clf=get_svm(**args)

    acc = train(clf, x=args['train_x'], y= args['train_y'], test_size=args['test_size'], max_iter= args['max_iter'])
    return json.dumps({'state': 'done', 'acc': acc})


@svm_api.route('/predict', methods=['post'])
def svm_predict():
    '''
    预测准确度
    Returns:
        预测结果
    '''
    args = {
        'predict_x': None
    }
    acq = ['predict_x']
    types = {
        'predict_x': [list, list, float]
    }

    args = get_json_data(raw=request.get_data(as_text=True), args=args)
    missed_arg = check_aquired_args(args=args, acqs=acq)
    if missed_arg is not None:
        return json.dumps({'state': 'fail', 'msg': '缺少必要参数'})
    args = type_format(args=args, types=types)

    clf = get_trained_svm()
    if clf is None:
        return json.dumps({'state': 'fail', 'msg': '模型尚未训练'})
    rs = predict(clf, args['predict_x'])
    return json.dumps({'state': 'done', 'result': rs.tolist()})




