from machine_learning.dl_method.BiGRUClassifier import BiGRUClassifier
from machine_learning.dl_method.BiLSTMClassifier import BiLSTMClassifier
from machine_learning.dl_method.CNNBiLSTM import CNNBiLSTM
from machine_learning.dl_method.CNNTransformer import CNNTransformer
from machine_learning.dl_method.CNNClassifier import ResCNNClassifier
from machine_learning.dl_method.TCNClassifier import TCNClassifier
from machine_learning.dl_method.TransformerClassifier import TransformerClassifier


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


methods = [
    ('BiGRUClassifier', BiGRUClassifier),
    ('BiLSTMClassifier', BiLSTMClassifier),
    ('ResCNNClassifier', ResCNNClassifier),
    ('CNNBiLSTM', CNNBiLSTM),
    ('CNNTransformer', CNNTransformer),
    ('TCNClassifier', TCNClassifier),
    ('TransformerClassifier', TransformerClassifier),
]

input_dim = 64
print(f"Checking models with input_dim = {input_dim}")
print("-" * 60)
print(f"{'Model Name':<25} | {'Parameters':<15}")
print("-" * 60)

for name, method in methods:
    try:
        model = method(input_dim)
        params = count_parameters(model)
        print(f"{name:<25} | {params:,}")
    except Exception as e:
        print(f"{name:<25} | Error: {e}")
print("-" * 60)
