import torch
import torchvision
import numpy as np
from torch import nn
import timm
import torch.nn.functional as F

device = 'cpu'

labels = np.array(['Your Eye Has Been Diagnosed With Cataracts', 'Your Eye Has Been Diagnosed With Glaucoma', 'Your Eye Is In A Good Condition! Keep It Up!', 'Your Eye Has Been Diagnosed With Uveitis'], dtype='str')
label2pred = dict(zip(labels, range(0, 4)))
ed_pred2label = dict(zip(range(0, 4), labels))
n_classes = len(np.unique(labels))

class ClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)          
        return loss, acc

    def validation_step(self, batch):
        images, labels = batch 
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)                    
        loss = F.cross_entropy(out, labels)  
        acc = accuracy(out, labels)          
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()    
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    model.train()
    return model.validation_epoch_end(outputs)

class EfficientNetB3(ClassificationBase):
    
    def __init__(self):
        super().__init__()
        
        self.network = timm.create_model('efficientnet_b3')
        num_ftrs = self.network.classifier.in_features
        self.network.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, n_classes))
        
        
    def forward(self, batch):
        batch = batch.to(device)
        return F.softmax(self.network(batch))

ed_test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((300, 300))
])

