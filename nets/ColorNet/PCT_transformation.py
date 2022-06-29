import torch
import torch.nn as nn
import numpy as np

'''
class ColorTransformation(nn.Module):
    def __init__(self, config):
        super(ColorTransformation, self).__init__()
        file = config['color_transformation_path']
        self.W1 = torch.tensor(np.load(file)["weight1"], dtype=torch.float32).cuda()
        self.W1 = self.W1.unsqueeze(0).unsqueeze(0)
        self.W2 = torch.tensor(np.load(file)["weight2"], dtype=torch.float32).cuda()
        self.W2 = self.W2.unsqueeze(0).unsqueeze(0)
        self.b = torch.tensor(np.load(file)["bias"], dtype=torch.float32).cuda()

    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.matmul(x.pow(2), self.W2) + torch.matmul(x, self.W1) + self.b
        x = x.transpose(1, -1)
        return x
'''

class PCTTransformation(nn.Module):
    def __init__(self, config):
        super(PCTTransformation, self).__init__()
#        use_cuda = config['cuda']
#        device_ids = config['gpu_ids']

        file = config['color_transformation_path']
        W1 = torch.tensor(np.load(file)["weight1"], dtype=torch.float32)
        self.W1 = torch.nn.Parameter(W1.unsqueeze(0).unsqueeze(0))
        W2 = torch.tensor(np.load(file)["weight2"], dtype=torch.float32)
        self.W2 = torch.nn.Parameter(W2.unsqueeze(0).unsqueeze(0))
        b = torch.tensor(np.load(file)["bias"], dtype=torch.float32)
        self.b = torch.nn.Parameter(b)

    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.matmul(x.pow(2), self.W2) + torch.matmul(x, self.W1) + self.b
        x = x.transpose(1, -1)
        x = torch.clamp(x, 0., 1.)
        return x

# linear PCT transformation
class PCTLinearTransformation(nn.Module):
    def __init__(self, config):
        super(PCTLinearTransformation, self).__init__()
#        self.use_cuda = use_cuda
#        self.device_ids = device_ids
        self.color_mapping = torch.nn.Parameter(torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])) # 3x3
    # transform the color
    def forward(self, x):
        # transform the input
        n, c, h, w = x.shape
        y = torch.matmul(self.color_mapping, x.view(n, c, -1))

        #y = torch.clamp(y, -1., 1.)
        y = torch.clamp(y, 0., 1.)
        return y.view(n, c, h, w)

# linear PCT transformation
class PCTLinearBiasTransformation(nn.Module):
    def __init__(self, config):
        super(PCTLinearBiasTransformation, self).__init__()
#        self.use_cuda = use_cuda
#        self.device_ids = device_ids
        self.color_mapping = torch.nn.Parameter(torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])) # 3x3
        self.b = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))

    # transform the color
    def forward(self, x):
        # transform the input
        n, c, h, w = x.shape
        y = torch.matmul(self.color_mapping, x.view(n, c, -1))
        #y += self.b.view(1, 3, 1)

        #y = torch.clamp(y, -1., 1.)
        min_y = torch.min(y)
        max_y = torch.max(y)
        y = (y -min_y ) / (max_y-min_y)
#       y = torch.clamp(y, 0.0, 1.)
        return y.view(n, c, h, w)

# non-linear PCT transformation
class PCTNeuralTransformation(nn.Module):
    def __init__(self, config):
        super(PCTNeuralTransformation, self).__init__()
#        self.use_cuda = use_cuda
#        self.device_ids = device_ids
        self.M1 = torch.nn.Parameter(torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])) # 3x3
        self.M2 = torch.nn.Parameter(torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])) # 3x3

    # transform the color
    def forward(self, x):
        # transform the input
        n, c, h, w = x.shape
        y = torch.matmul(self.M1, x.view(n, c, -1))
        y = nn.functional.relu(y)
        y = torch.matmul(self.M2, y.view(n, c, -1))
        y = torch.clamp(y, 0., 1.)
#        y = torch.clamp(y, 0.0, 1.)
        return y.view(n, c, h, w)

'''
# non-linear PCT transformation
class PCTNeuralTransformation(nn.Module):
    def __init__(self, config):
        super(PCTNeuralTransformation, self).__init__()
#        self.use_cuda = use_cuda
#        self.device_ids = device_ids
        fc_dim = 100
        self.fc_transform = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(3, fc_dim),
            nn.ReLU(True),
            nn.Linear(fc_dim, 3)
        )

    # transform the color
    def forward(self, x):
        # transform the input
        n, c, h, w = x.shape
        output = self.fc_transform(x.view(-1,c))
        #output = torch.tanh(output)
        #output = 0.5* (output + 1.0) # normalize to [0 1]
        #output = torch.clamp(output, 0.0, 1.0)
        return output.view(n,c,h,w)
'''

class PCTTransformationOld2New(nn.Module):
    def __init__(self):
        super(PCTTransformationOld2New, self).__init__()
#        use_cuda = config['cuda']
#        device_ids = config['gpu_ids']

        file = 'weights2_old2new_.npz'
        W1 = torch.tensor(np.load(file)["weight1"], dtype=torch.float32)
        self.W1 = torch.nn.Parameter(W1.unsqueeze(0).unsqueeze(0))
        print (self.W1)
        W2 = torch.tensor(np.load(file)["weight2"], dtype=torch.float32)
        self.W2 = torch.nn.Parameter(W2.unsqueeze(0).unsqueeze(0))
        print (self.W2)
        b = torch.tensor(np.load(file)["bias"], dtype=torch.float32)
        self.b = torch.nn.Parameter(b)

    def forward(self, x):
        x = x.transpose(1, -1)
        x = torch.matmul(x.pow(2), self.W2) + torch.matmul(x, self.W1) + self.b
        x = x.transpose(1, -1)
        return x

