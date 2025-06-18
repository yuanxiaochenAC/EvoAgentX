from ..base import Predictor


def aggregate(n_predictors: int = 3):
    """
    聚合多个predictor的结果
    
    Args:
        n_predictors: predictor的数量，从1,3,5,7,9中选择，默认为3
    """
    # 验证n_predictors参数
    valid_predictors = [1, 3, 5, 7, 9]
    if n_predictors not in valid_predictors:
        raise ValueError(f"n_predictors必须是{valid_predictors}中的一个值，当前值为{n_predictors}")
    
    pass