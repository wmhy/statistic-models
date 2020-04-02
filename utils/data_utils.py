import json


def type_format(args, types):
    '''
    将按照传入的格式进行转换，
    仅仅转换目标类型为list, int, float, boolean的， string类型和其他类型不转换
    Args:
        args: 字典，每个key表示需要转换的元素
        types: 字典，其中每个key对应的value表示args中key元素对应的类型，不在type中的key将不进行转换

    Returns:

    '''
    for k in types.keys():
        arg_v = args[k]
        #对空值不做转换
        if arg_v is None:
            continue
        #去除首尾空格
        if type(arg_v) == str:
            arg_v = arg_v.strip()

        if type(types[k]) == str:
            types[k] = types[k].strip()
        t = types[k]
        #目标类型为列表时，其对应的type也为列表,且源类型为string，只需要进行一次转换就能将所有嵌套列表转换出来,此处不转换其中的元素
        if type(t) == list:
            if type(arg_v)== str:
                arg_v =json.loads(arg_v)

        else:
            if t == int:
                arg_v = int(arg_v)
            elif t == float:
                arg_v = float(arg_v)
            elif t == bool:
                arg_v = True if (arg_v == 'True') else False

        args[k] = arg_v

    return args


    return None;


def get_json_data(raw, args):
    '''
    将raw字符串转化为targ中的数据
    Args:
        raw: json字符串
        args: 字典，其中的键表示希望存在的值
        types: 值的类型

    Returns:
    当字符串不能被转换成json时返回None
    否则尝试将targ中的每一个键，在raw中获取同名键对应的值
    '''
    try:
        json_data = json.loads(raw)
    except ValueError as e:
        return None

    for k in args.keys():
        if k in json_data:
            #不为空才赋值
            if json_data[k] != '':
                args[k] = json_data[k]

    return args

def check_aquired_args(args, acqs):
    '''
    检查必须参数是否有值，不进行去除首尾空格
    Args:
        args: 参数表，字典
        acqs: 必须参数名，字典

    Returns:
    None: 符合
    否则返回第一个不符合的参数名
    '''
    for i in acqs:
        if i not in args:
            return i
        if (args[i] is None) or (args[i] == []) or (args[i] == ''):
            return i
    return None

