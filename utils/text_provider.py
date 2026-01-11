'''
文本获取模块
获取问答模板
获取对应的类映射
'''
'''
尝试先使用brat脑瘤数据集建立相关模板
标签描述

'''

import random

# 设计随机种子


qva_template1 = 'Find nodules enlarged by <n>'
qva_template2 = 'Locate nodules that have shrunk by <n> in volume'
qva_template3 = 'Find newly appeared nodules'

def generate_qva(seed):
    random.seed(seed)
    rand_num = random.randint(5, 40)
    qva_template = random.choice([qva_template1, qva_template2, qva_template3])
    if(qva_template == qva_template3):
        qva_num = 3
    elif(qva_template == qva_template1):
        qva_num = 1
    else:
        qva_num = 2
    qva_template = qva_template.replace('<n>', str(rand_num))
    return qva_template, qva_num