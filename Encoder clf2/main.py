
import  argparse
from train import training, test

def main(args):

    if args.training:
        training(args)
    if args.test:
        test(args)

if __name__=='__main__':
    
    parser= argparse.ArgumentParser(description='Parsing Method')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--h', default=8, type=int)
    parser.add_argument('--d_ff', default=2048)
    parser.add_argument('--N', default=6, type=int)
    parser.add_argument('--dropout', default=0.1 ,type=int)

    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--maxlen', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    args= parser.parse_args()


    main(args)












