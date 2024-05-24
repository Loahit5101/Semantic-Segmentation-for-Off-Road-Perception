import os
import torch
import yaml
import matplotlib.pyplot as plt
from utils import get_dataloader

if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataloaders = get_dataloader(config['DATA_PATH'],
                                 batch_size=config['BATCH_SIZE'],
                                 resize_shape=(config['IMG_HEIGHT'], config['IMG_WIDTH']))
    
   
    for batch in dataloaders['train']:
        inputs, labels = batch['image'], batch['mask']
        
        # Visualize each sample in the batch
        for i in range(len(inputs)):
            image, mask = inputs[i], labels[i]
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 9, 1)
            plt.imshow(image.permute(1, 2, 0).cpu().numpy())  
            plt.title('Image')
            plt.axis('off')
            
            for j in range(mask.shape[0]):
                plt.subplot(1, 9, j+2)
                plt.imshow(mask[j].cpu().numpy(), cmap='gray')  
                plt.title(f'Mask Channel {j+1}')
                plt.axis('off')
                
            plt.show()

