import torch
import torchxrayvision as xrv

class Network(torch.nn.Module):
    def __init__(self, weights='all', dropout_prob=0):
        super(Network, self).__init__()
        self.base_model = xrv.models.DenseNet(weights=weights)
        self.dropout = torch.nn.Dropout(p=dropout_prob) if dropout_prob else None
        num_in = self.base_model.features.norm5.bias.shape[0]
        self.base_model.classifier = torch.nn.Identity() # remove pretrained FC layer
        self.base_model.op_threshs = None # turn off output normalization
        self.appear_fc = torch.nn.Linear(in_features=num_in, out_features=4)
        self.grade_fc = torch.nn.Linear(in_features=num_in, out_features=3)
    
    def forward(self, input, out_type):
        x = self.base_model(input)
        if self.dropout: x = self.dropout(x)

        if out_type == 'both':
            return self.appear_fc(x), self.grade_fc(x)
        elif out_type == 'appear':
            return self.appear_fc(x), None
        elif out_type == 'grade':
            return None, self.grade_fc(x)
        else:
            raise ValueError("output type must be one of 'appear', 'grade', or 'both'")

    def freeze(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for name, param in self.base_model.named_parameters():
            if not 'norm' in name:
              param.requires_grad = True
