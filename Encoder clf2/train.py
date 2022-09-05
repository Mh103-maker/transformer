import torch
import torch.nn as nn
from torch.nn import functional 
from torch.utils.data import DataLoader
import time
from model.encoderclf import Encoderclf
from dataset import create_dataloader


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model.generator(model(text.to(device)))
            label = label.type(torch.LongTensor)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

def training(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    #data
    train_dataloader, valid_dataloader,test_dataloader, vocab_len= create_dataloader(args.batchsize, args.maxlen)

    #model init
    model=Encoderclf(src_vocab=vocab_len,n_class=2, d_model=args.d_model, h=args.h, d_ff=args.d_ff,N=args.N, dropout=args.dropout )

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
        epoch_start_time= time.time()

        model.train()
        total_acc, total_count = 0, 0
        log_interval = 100
        start_time = time.time()
        for idx, (label, text) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predicted_label = model.generator(model(text.to(device)))
            label = label.type(torch.LongTensor).to(device)
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
                    '|loss {:8.3f}| accuracy {:8.3f}'.format(epoch, idx, len(train_dataloader),total_loss.item(),
                                                total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text) in enumerate(valid_dataloader):
                predicted_label = model.generator(model(text.to(device)))
                label = label.type(torch.LongTensor).to(device)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        accu_val=total_acc/total_count


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
    ######test
    #    
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(test_dataloader):
            predicted_label = model.generator(model(text.to(device)))
            label = label.type(torch.LongTensor).to(device)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    acc= total_acc/total_count
    print('test accurancy: {:8.3f}'.format(acc))
