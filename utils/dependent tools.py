'''
辅助函数，与原项目无关
'''

import json

def get_attr_from_str(str):
    '''
    从python的默认函数参数中获取属性，例如
    "C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,probability=False, "
                      "tol=0.001,cache_size=200,class_weight=None, verbose=False, max_iter=-1,decision_function_shape='ovr', "
                      "random_state=None"
    得到
    'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'auto', 'coef0': 0.0, 'shrinking': True, 'probability': False,
    'tol': 0.001, 'cache_size': 200, 'class_weight': None, 'verbose': False, 'max_iter': -1, 'decision_function_shape':
    'ovr', 'random_state': None,
    Args:
        str:

    Returns:

    '''
    s = str.split(',')
    s = [i.strip() for i in s]

    #输出属性和默认值以便创建字典接收属性
    print('attr and vals:')
    for i in s:
        k, v = i.split('=')
        k = k.strip().strip()
        print("'"+k+"': ", end='')
        print(v, end=', ')
    print()
    print('attrs:')
    #输出属性名以便填充函数形参
    for i in s:
        k, _ = i.split('=')
        k = k.strip()
        print(k, end=', ')

    #输出属性赋值，以便填充函数调用
    print()
    print('attr assgin:')
    for i in s:
        k, _ = i.split('=')
        k=k.strip()
        print(k+'='+k, end=', ')
    #print(datas)

    #输出推断类型，以便进行类型检查,只推断了float, int str三种，默认str
    print()
    print('type assgin:')
    for i in s:
        k, v = i.split('=')
        types = 'str'
        if '.' in v:
            types = 'float'
        elif v.isdigit():
            types = 'int'
        print("'"+k+"': "+types, end=', ')



if __name__ == '__main__':
    get_attr_from_str("C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,probability=False, "
                      "tol=0.001,cache_size=200,class_weight=None, verbose=False, max_iter=-1,decision_function_shape='ovr', "
                      "random_state=None")