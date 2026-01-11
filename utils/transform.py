from monai.transforms import MapTransform
import torch
import numpy as np
from models.text_processor import LanguageProcessor
from utils.text_provider import generate_qva
import re
class ProcessText(MapTransform):
    def __init__(self,language_processor: LanguageProcessor, keys: list, num_samples: int = 1,max_len: int = 60):
        super().__init__(keys)
        self.max_len = max_len
        self.language_processor = language_processor
        self.num_samples = num_samples

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.language_processor.tokenizer.tokenize(d[key])
            d[key] = self.language_processor.tokenizer.convert_tokens_to_ids(d[key])
            # 将token_ids填充到max_len
            d[key] = d[key] + [0] * (self.max_len - len(d[key]))
            
            # 转换为torch tensor
            # d[key] = torch.tensor(d[key])
             
        return d #这里就是返回一个但采样的textids,在后续再进行多采样工作

class ProcessLabel(MapTransform):
    def __init__(self, keys: list):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)# 在这里阅读对应的
        text = d['text']
        def get_kind(text):
            '''
            目前就将其分为三中情况
            1.肺结节增长了n
            2.肺结节减少了n
            3.肺结节新增了
            '''
            kind = rate = 0
            if "enlarged" in text:
                kind = 1
                # 正则化提取数字
                numbers = re.findall(r'\d+', text)
                if numbers:
                    rate = float(numbers[0])
            elif "shrunk" in text:
                kind = 2
                # 正则化提取数字
                numbers = re.findall(r'\d+', text)
                if numbers:
                    rate = float(numbers[0])
            else:
                kind = 3
            return kind, rate
        kind, rate = get_kind(text)

        def get_nodules(list,kind, rate):
            items = []
            for item in list:
                if kind ==1:
                    if 'Growth' in item and float(item.strip().split(' ')[-1]) >= rate:
                        items.append(item)
                elif kind ==2:
                    if 'Reduction' in item and float(item.strip().split(' ')[-1]) >= rate:
                        items.append(item)
                else:
                    if 'New' in item:
                        items.append(item)
            return items
        nodules = get_nodules(d['label'],kind, rate)
        path = d['label_image']
        # 1. 读取图像和元数据
        itk_img = sitk.ReadImage(path)
        origin = np.array(itk_img.GetOrigin())   # (x, y, z) 世界坐标原点
        spacing = np.array(itk_img.GetSpacing()) # (x, y, z) 像素间距
        #nodules对应的像素坐标为[(x_pix, y_pix, z_pix, d_pix), ...]
        nodules_pix = []
        for nodule in nodules:
            x_mm, y_mm, z_mm, d_mm = map(float, nodule[1:])
            x_pix = int((x_mm - origin[0]) / spacing[0])
            y_pix = int((y_mm - origin[1]) / spacing[1])
            z_pix = int((z_mm - origin[2]) / spacing[2])
            d_pix = int(d_mm / spacing[2]) # 假设d是直径，取z轴方向的间距
            nodules_pix.append((x_pix, y_pix, z_pix, d_pix))

        

        
        return d