import torch
from torch.nn.functional import pad
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator


tokenizer=get_tokenizer('basic_english')
train_iter = IMDB(split='train')

# tokening
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text) 

# vocabulary (train set 에 대해서만)
vocab_src = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>",'<blank>'])
vocab_src.set_default_index(vocab_src["<unk>"])

# pipeline
text_pipeline = lambda x: vocab_src(tokenizer(x)) # vocabulary 내 word 위치
label_pipeline = lambda x: 1. if (x=='pos') else 0 # pos==1, neg==0

def collate_batch(

     batch,
     max_padding,
     pad_id,
):
    label_list, text_list = [], []
    for (_label, _text) in batch:
         label_list.append(torch.tensor(label_pipeline(_label), dtype=torch.int))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int )
         text_list.append(
               pad(
                processed_text,
                (
                    0,
                    max_padding - len(processed_text),
                ),
                value=pad_id,
               )
          )
    src = torch.stack(text_list)
    tgt = torch.stack(label_list)
    return (tgt, src)


def dataloader(
    batch_size,
    max_padding,
):
    def collate_fn(batch):
        return collate_batch(
            
            batch,
            text_pipeline,
            label_pipeline,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()['<blank>'],
        )

    train_iter, test_iter=IMDB(split=('train','test'))

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    num_train = int(len(train_iter_map) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_iter_map, [num_train, len(train_iter_map) - num_train])
    
    test_iter_map=to_map_style_dataset(test_iter)

    test_dataloader = DataLoader(
        test_iter_map, 
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collate_fn,
        )

    train_dataloader = DataLoader(
        split_train_,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_dataloader = DataLoader(
        split_valid_,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    return train_dataloader, valid_dataloader, test_dataloader, len(vocab_src)