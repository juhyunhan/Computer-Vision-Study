import argparse
import torch
import torch.nn.functional as F

from utils.train_util import get_log, create_save_dir 
from data.cifar10 import create_dataloader
from models.hot_model import HotModel

def parse_args():
    parser = argparse.ArgumentParser(description='train, validation test argument')
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--project_name', '-n', type=str, default="default")
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001)
    parser.add_argument('--print_freq', '-pf', type=int, default=100)

    
    return parser.parse_args()


def train(train_loader, model, optimizer, criterion, print_freq, logger):
    model.train()
    
    total = 0
    correct = 0

    for i, (img, label) in enumerate(train_loader):
        total += img.shape[0]
        pred = model(img)
        logits = F.log_softmax(pred, dim=1)
        loss = criterion(logits, label)   
        
        loss.backward()
        optimizer.step()
        
        pred_class = torch.argmax(logits.detach(), dim = 1)
        
        for num in range(img.shape[0]):
            if label[num] == pred_class[num]:
                correct += 1
                
        accuracy = correct / total * 100
        
        
        if i % print_freq == 0:
            logger.info(f" [LOSS]] : {loss.item()} | [ACC]:{accuracy}")
        

            
def validate(val_loader, model, criterion, print_freq, logger):
    model.eval()
    total = 0
    correct = 0
    
    for i, (img, label) in enumerate(val_loader):
        total += 1
        pred = model(img)
        logits = F.log_softmax(pred, dim=1)
        loss = criterion(logits, label)   
        
        loss.backward()
        
        #! metrics
        pred_class = torch.argmax(logits, dim = 1)
        if label == pred_class:
            correct += 1
        accuracy = correct / total * 100
        logger.info(f"accuracy : {accuracy}")
        if i % print_freq == 0:
            logger.info(f"loss : {loss.item()}")


def main(args):
    #epoch = args.epochs
    #batch_size = args.batch_size
    path = create_save_dir(name =  args.project_name)
    logger = get_log(path= path, mode= 'train')
    
    logger.info(args)
    
    #! make data loader
    train_loader, val_loader = create_dataloader(batch_size=args.batch_size)
    
    #! make model
    model = HotModel()
    
    #! make optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    #! make criterion (Loss function)
    criterion = torch.nn.CrossEntropyLoss()    
    
    total_epoch = args.epochs #30
    for epoch in range(total_epoch):
        train(train_loader, model, optimizer, criterion, args.print_freq, logger)
        #validate()
    
if __name__ == '__main__':
    args = parse_args()
    main(args)