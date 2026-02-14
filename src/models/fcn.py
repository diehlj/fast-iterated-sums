import torch
import torch.nn as nn

def get_num_layer_for_fcn(var_name):
    if 'fc1' in var_name:
        return 0
    elif 'fc2' in var_name:
        return 1
    elif 'fc3' in var_name:
        return 2
    else:
        return 3

class FCN(nn.Module):
    def __init__(self, num_classes=3, input_channels=3, input_height=32, input_width=32, 
                 hidden_size=64, dropout_rate=0.5):
        super(FCN, self).__init__()
        
        input_size = input_channels * input_height * input_width

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # self.register('fc2.weight'
        setattr(self.fc2.weight, "_optim", {"weight_decay": 0.00008, "lr": 0.88})
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    #def register(self, name, tensor, lr=None):
    #    """Register a tensor with a configurable learning rate and 0 weight decay"""

    #    if lr == 0.0:
    #        self.register_buffer(name, tensor)
    #    else:
    #        self.register_parameter(name, nn.Parameter(tensor))

    #        optim = {"weight_decay": 0.0}
    #        if lr is not None: optim["lr"] = lr
    #        setattr(getattr(self, name), "_optim", optim)