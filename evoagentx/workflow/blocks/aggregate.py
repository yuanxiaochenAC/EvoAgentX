import re
import string
import unicodedata
from collections import Counter
from evoagentx.workflow.blocks.block import block
from evoagentx.workflow.operators import Predictor

class aggregate(block):
    
    def __init__(self, llm, n=3):
        self.predictor = Predictor(llm=llm)
        self.n = n
        self.activate = True
    
    def __call__(self, question, **kwargs):
        """聚合多个预测结果，返回最常见的答案"""
        predictions = []
        
        # 生成n个预测
        for _ in range(3):
            prediction = self.predictor.execute(question=question)
            predictions.append(prediction['answer'])
        
        # 标准化并统计
        normalized_predictions = [self._normalize_text(answer) for answer in predictions]
        normalized_predictions = [x for x in normalized_predictions if x is not None]
        
        if not normalized_predictions:
            return predictions[0] if predictions else None
            
        # 找到最常见的标准化答案
        value_counts = Counter(normalized_predictions)
        most_common_normalized = value_counts.most_common(1)[0][0]
        
        # 返回对应的原始答案
        for prediction in predictions:
            if self._normalize_text(prediction) == most_common_normalized:
                return prediction
        
        return predictions[0]  # 兜底返回第一个答案

    def execute(self, question):
        """执行预测并返回所有结果"""
        predictions = []
        
        for _ in range(self.n):
            prediction = self.predictor.execute(question=question)
            predictions.append(prediction['answer'])

        return predictions
    
    def _normalize_text(self, text):
        """标准化文本用于比较"""
        if not text:
            return None
            
        # Unicode标准化
        text = unicodedata.normalize("NFD", text)
        
        # 转小写
        text = text.lower()
        
        # 移除标点符号
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)
        
        # 移除冠词
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        
        # 清理空白字符
        text = " ".join(text.split())
        
        return text