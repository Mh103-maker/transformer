import torch
import torch.nn as nn
from torch.nn import functional 
from torch.utils.data import DataLoader
import time
from model.encoderclf import Encoderclf
from .dataset import create_dataloader

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model.generator(model(text))
        label = label.type(torch.LongTensor)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_loss=loss
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '|loss {:8.3f}| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),total_loss.item()
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model.generator(model(text))
            label = label.type(torch.LongTensor)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

def training(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #data
    train, valid,_, vocab_len= create_dataloader(args.batchsize, args.maxlen)

    #model init
    model=Encoderclf(src_vocab=vocab_len,n_class=2, d_model=args.d_model, h=args.h, d_ff=args.d_ff, dropout=args.dropout )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model= model.to(device)

    #optimizer LR
    criterion = torch.nn.CrossEntropyLoss()
    lr = args.lr# learning rate
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu=0.6

    #train
    for epoch in range(1, args.epochs+1):
        epoch_start_time= time.time
        train(train)
        accu_val = evaluate(valid)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} '.format(epoch,
                                                time.time() - epoch_start_time,
                                                accu_val))
            print('-' * 59)


def test(args): 
    _,_,test,_=create_dataloader(args.batchsize, args.maxlen)
    accu_test=evaluate(test)
    print('test accuracy {:8.3f}'.format(accu_test))

