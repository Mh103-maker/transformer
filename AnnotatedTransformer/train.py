import torch
from Model.Transformer import make_model
from dataset import build_vocabulary, create_dataloaders
import time

def train(model, dataloader, optimizer):

    model.train()
    start_time=time.time()



def eval(model, dataloader):
    model.eval()



def training(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    vocab_src, vocab_tgt= build_vocabulary()

    model=make_model(
        len(vocab_src), len(vocab_tgt), 
        N=args.N, d_model=args.d_model,
        d_ff=args.d_ff, h=args.h, dropout=args.dropout)

    model=model.to(device)

    criterion=torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    train_dataloader, valid_dataloader, test_dataloader= create_dataloaders(
        device=device, 
        batch_size=args.batchsize,
        max_padding=args.maxlen,
    )

    for epoch in range(1, args.epochs+1):
        train(model, train_dataloader, optimizer=optimizer)



