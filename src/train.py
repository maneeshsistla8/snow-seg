import torch
from torch.optim import Adam
from torch import nn

from datasets import preprocess_train_data, load_train_data
from backboned_unet import Unet

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
input_rasters_path = '../data/train/Rasters/Harmonized/'
labels_path = '../data/train/Labels/Harmonized/'
model_save_path = '../weights/model_weights_harmonized.pt'

def adjust_learning_rate_poly(optimizer, iteration):
    if iteration <= 60:
       lr = 5.0e-4
    else:
       lr = 1.0e-4

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def main():
    print(f'Device: {device}')

    model = Unet(backbone_name='resnet50', pretrained=True, encoder_freeze=True, classes=2)
    model = model.to(device)

    LEARNING_RATE = 5e-4
    num_epochs = 200
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    dataloader, num_samples = load_train_data(input_rasters_path, labels_path)

    model.train()
    for epoch in range(num_epochs):
        print("Now in Epoch =>", str(epoch))
        adjust_learning_rate_poly(optimizer, epoch)
        epoch_loss = 0
        for item in dataloader:
            item = preprocess_train_data(item)
            X = item['image']
            y = item['mask']
            X = X.to(device)
            y = y.to(device)
            prediction = model(X)
            y = torch.squeeze(y,1)
            y = y.to(device)
            
            loss = loss_fn(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.item()

        if epoch % 10 == 0 or epoch == num_epochs-1:
            path = f'../checkpoints/checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, path)

        print("Average loss in this epoch is "+str(epoch_loss/num_samples))

    print(f'Saving model to path: {model_save_path}')
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    main()