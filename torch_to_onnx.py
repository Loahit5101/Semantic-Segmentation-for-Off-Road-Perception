import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import yaml

with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)
num_classes = config['NUM_CLASSES']
model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
state_dict = torch.load("best_model.pth")
model.load_state_dict(state_dict)
torch_input = torch.randn(1, 3, 544, 1024) # Input Image size, batch = 1
torch.onnx.export(model, torch_input, "model.onnx", verbose=True)

