import torch
import torch.nn as nn
import torchvision.models as models



class STModel(nn.Module):
    def __init__(self, pretrained=True, trainable=True, gene_num=1000, dropout=0.):
        super().__init__()

        # 加载预训练的 DenseNet-121 模型
        self.model = models.densenet121(pretrained=pretrained)
        self.model.classifier = nn.Linear(in_features=1024, out_features=gene_num, bias=True)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        x = self.model(x)
        return x