import argparse
import torch
import torch.nn.functional as F
import wandb
import torchvision

from models.resnet import ResNet
from utils.train_util import get_log, create_save_dir 
from data.cifar10 import create_dataloader
from models.hot_model import HotModel

def parse_args():
    parser = argparse.ArgumentParser(description='train, validation test argument')
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--project_name', '-n', type=str, default="default")
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--print_freq', '-pf', type=int, default=100)

    
    return parser.parse_args()


def train(train_loader, model, optimizer, criterion, print_freq, logger):
    model.train()
    
    total = 0
    correct = 0

    for i, (img, label) in enumerate(train_loader):
        if i > 3000:
            break
        total += img.shape[0]
        pred = model(img)
        logits = F.log_softmax(pred, dim=1) #predict 한 값을 확률로 바꾸어 주는 것
        loss = criterion(logits, label)   #coss entropy 1인 부분만 확률이 올라가게 만들어 주는것 
        
        loss.backward() #미분하면서 gradient 값이 생김
        optimizer.step() # weight 값이 update 됨
        
        pred_class = torch.argmax(logits.detach(), dim = 1) #가장 확률이 높은것에 대해서 index를 뽑은 것
        
        for num in range(img.shape[0]):
            if label[num] == pred_class[num]: 
                correct += 1 
                
        accuracy = correct / total * 100 #정답률을 구한 것.
        
        
        
        if i % print_freq == 0:
            wandb.log({"Train/Accuracy" : accuracy,
                       "Train/Loss" : loss.item()})
            logger.info(f" [LOSS]] : {loss.item()} | [ACC]:{accuracy}")
        

            
def validate(args, val_loader, model, criterion, print_freq, logger):
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for i, (img, label) in enumerate(val_loader):
            total += img.shape[0]
            pred = model(img)
            logits = F.log_softmax(pred, dim=1)
            loss = criterion(logits, label)   
            
            pred_class = torch.argmax(logits.detach(), dim = 1)
            #! metrics
            for num in range(img.shape[0]):
                if label[num] == pred_class[num]: 
                    correct += 1 
                
            accuracy = correct / total * 100 
            logger.info(f"accuracy : {accuracy}")
            if i % print_freq == 0:
                logger.info(f" [LOSS]] : {loss.item()} | [ACC]:{accuracy}")
                
        wandb.log({"Val/Accuracy" : accuracy})

def main(args):
    #epoch = args.epochs
    #batch_size = args.batch_size
    path = create_save_dir(name =  args.project_name)
    logger = get_log(path= path, mode= 'train')
    
    logger.info(args)
    wandb.init(project = 'DLStudy', name = args.project_name)
    wandb.config.update(args)
    
    #! make data loader
    train_loader, val_loader = create_dataloader(batch_size=args.batch_size)
    
    #! make model
    #model = HotModel()
    model = ResNet([2,2,2], num_classes=10)
    wandb.watch(model)
    
    
    #! make optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    #! make criterion (Loss function)
    criterion = torch.nn.CrossEntropyLoss()    
    
    total_epoch = args.epochs #30
    for epoch in range(total_epoch):
        train(train_loader, model, optimizer, criterion, args.print_freq, logger)
        validate(args, val_loader, model, criterion, args.print_freq, logger)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)