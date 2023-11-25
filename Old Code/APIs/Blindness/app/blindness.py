import torch
import torchvision
import numpy as np
from torch import nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

device='cpu'

un_labels = np.array(['You Are Not At Risk Of Blindness. No DR!', 'You Have Mild DR. Be Careful!', 'You Have Moderate DR. Consult An Opthalmologist On Future Action', 'You Have Severe DR. Seek Help Immediately!', 'You Have Profilerative DR! You Are At High Risk Of Blindness; Take Immediate Action!'], dtype='str')
label2pred = dict(zip(un_labels, range(0, 5)))
pred2label = dict(zip(range(0, 5), un_labels))
n_classes = len(np.unique(un_labels))

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
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}\n".format(epoch, result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

class EfficientNetB4(ClassificationBase):
    
    def __init__(self):
        super().__init__()
        
        self.network = EfficientNet.from_pretrained('efficientnet-b4')
        self.network._fc = nn.Linear(1792, n_classes)
        
    def forward(self, batch):
        batch = batch.to(device)
        return self.network(batch)

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((400, 400)),
    torchvision.transforms.CenterCrop((380)),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
