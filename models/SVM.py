
#svm有关
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

#模型
clf = None


def get_svm(C, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight,
            verbose, max_iter, decision_function_shape, random_state,
            train_x=None, train_y=None, test_size=0):
    '''

    Args:
        C: 错误项的惩罚系数，越大对误判样本惩罚越大，训练集上准确率越高，但是泛化能力减弱；减小c泛化能力加强，但在训练集上的准确率降低
        kernel: 采用核函数的类型可选的有：linear--线性核函数 ;poly--多项式核函数；rbf--径向核函数/高斯函数；sigmod:sigmod核函数；precomputed:矩阵函数
        max_iter: 最大迭代系数，-1表示不限制
        kernel:
        degree:
        gamma:
        coef0:
        shrinking:
        probability:
        tol:
        cache_size:
        class_weight:
        verbose:
        max_iter:
        decision_function_shape:
        random_state:
        train_x: 无效
        train_y: 无效
        test_size: 无效
    Returns:
        新建的SVM模型
    '''
    global clf
    clf = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                  shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size,
                  class_weight=class_weight, verbose=verbose, max_iter=max_iter,
                  decision_function_shape=decision_function_shape, random_state=random_state)
    return clf


def get_trained_svm():
    '''
    Returns
    -------
    已经训练好的SVM模型，没有时返回None
    '''
    global clf
    return clf


def train(clf_t, x, y, test_size, max_iter):
    '''

    Args:
        clf_t: 模型，不能为空
        x: 训练的x
        y: 训练的y
        test_size: 测试集占比
        max_iter: 最大迭代步数，默认为-1，即不限制

    Returns:
        acc: 准确率
    '''

    '''
    训练模型'''
    if (test_size > 0) and (test_size<1):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)
    #global clf
    '''
    if clf is None:
        clf=getSVM(C=c, kernel=kernel, max_iter=max_iter)
        '''

    clf_t.fit(train_x, train_y)
    prediction = clf_t.predict(test_x)
    acc = metrics.accuracy_score(prediction, test_y)
    return acc


def predict(clf_t, x):
    '''
    按照训练好的模型进行预测
    Args:
        clf: 模型，不能为空
        x: 需要预测的数据，为数组

    Returns:
    预测后的分类
    '''
    return clf.predict(x)