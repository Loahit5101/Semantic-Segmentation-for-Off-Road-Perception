import torch
import torch.nn as nn
import torchvision.models.segmentation as models
import yaml

'''
Class to edit the model ste dict to remove auxiliary output and output argmax(output['out'])
'''
class FCN_ResNet50(nn.Module):
    def __init__(self, model_path, num_classes=8):
        super(FCN_ResNet50, self).__init__()
       
        self.model = models.fcn_resnet50(pretrained=False, num_classes=num_classes)

        state_dict = torch.load(model_path)
        
        # Filter out 'aux_classifier' keys if they exist in the state_dict
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_classifier')}
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, inputs):
        x = self.model(inputs)['out']
        x = x.argmax(1, keepdim=True)
        return x

model = FCN_ResNet50('Best_model_so_far.pth', num_classes=8)

model.eval()

torch.save(model.state_dict(), 'fine_tuned_fcn_resnet50.pth')

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

torch_input = torch.randn(1, 3, 544, 1024)  # Example input 


print("Model exported to ONNX format as 'model2.onnx'")
torch.onnx.export(model, torch_input, "model2.onnx",
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    verbose=False)


