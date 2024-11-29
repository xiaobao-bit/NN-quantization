import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization.model_quantizer import model_quantizer
from utils.utils import model_test

# def activation_quan_hook_mm(module, input, output):
#     return asym_quantizer(input, 8, 1, 'Min_max')
#
# def activation_quan_hook_mse(module, input, output):
#     return asym_quantizer(input, 8, 1, 'MSE')

model_path = "model/pt_model/FashionMNIST_cnn_50.pt"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

model = Net()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
test_loss, accuracy = model_test(model)

model_mm_quan = Net()
test_loss_mm, accuracy_mm = model_quantizer(model_mm_quan, 8, 1, 'MM', model_path, 'A')

model_mse_quan = Net()
test_loss_mse, accuracy_mse = model_quantizer(model_mse_quan, 8, 1, 'MSE', model_path, 'A')

model_ce_quan = Net()
test_loss_ce, accuracy_ce = model_quantizer(model_ce_quan, 8, 1, 'CE', model_path, 'A')

print(f"============== Report ==============")
print(f"Ori model: Loss is: \t{test_loss}\tAccuracy is: \t{accuracy}%\n")

print(f"M-M model: Loss is: \t{test_loss_mm}\tAccuracy is: \t{accuracy_mm}%\n")

print(f"MSE model: Loss is: \t{test_loss_mse}\tAccuracy is: \t{accuracy_mse}%\n")

print(f" CE model: Loss is: \t{test_loss_ce}\tAccuracy is: \t{accuracy_ce}%\n")