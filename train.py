import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import yaml
import matplotlib.pyplot as plt
from utils import get_dataloader

def mean_iou(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    outputs = outputs.byte()
    labels = labels.byte()
    intersection = torch.logical_and(labels, outputs)
    union = torch.logical_or(labels, outputs)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score.mean().item()  # Convert to float

def train(model, dataloaders, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    train_losses, val_losses = [], []
    train_mious, val_mious = [], []
    best_val_miou = 0.0  
    best_model_wts = None  

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_miou = 0.0

            for batch in dataloaders[phase]:
                inputs = batch['image'].to(device)
                print(inputs.shape)
                labels = batch['mask'].to(device)

                label = torch.argmax(labels, dim=1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                miou = mean_iou(torch.argmax(outputs['out'], 1), label)
                running_miou += miou

                print(f'Batch loss: {loss.item()}, Batch mIoU: {miou}')

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_miou = running_miou / len(dataloaders[phase])

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_mious.append(epoch_miou)
            else:
                val_losses.append(epoch_loss)
                val_mious.append(epoch_miou)

                # Check if the current validation mIoU is the best so far
                if epoch_miou > best_val_miou:
                    best_val_miou = epoch_miou
                    best_model_wts = model.state_dict()  # Save the best model weights

            print(f'{phase} Loss: {epoch_loss:.4f}, mIoU: {epoch_miou:.4f}')

    # Plot the training and validation loss and mIoU
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_mious, label='Train mIoU')
    plt.plot(val_mious, label='Validation mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.title('mIoU over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Best model saved with mIoU: {best_val_miou:.4f}')

    return model

if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataloaders = get_dataloader(config['DATA_PATH'],
                                 batch_size=config['BATCH_SIZE'],
                                 resize_shape=(config['IMG_HEIGHT'], config['IMG_WIDTH']))

    model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)

    num_classes = config['NUM_CLASSES']
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    fine_tuned_model = train(model, dataloaders, criterion, optimizer, num_epochs=config['NUM_EPOCHS'])

