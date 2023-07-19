import argparse
from graph_builder import kddcup_graph, naver_graph
from utils import get_logger
from trainer import train

def main(args):
    logger = get_logger(name=f"{args.dataset}_{args.model}_{args.num_layers}_{args.batch_size}_{args.dropout}", path=f"{args.log_dir}/{args.dataset}_{args.model}_{args.num_layers}_{args.batch_size}_{args.dropout}.log")
    logger.info('train args')
    logger.info(args)

    if args.dataset == 'kddcup':
        dataset = kddcup_graph()
    else:
        dataset = naver_graph()
    train(args, dataset, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='kddcup', help="default dataset is kddcup15")
    parser.add_argument("--hidden_dim", type=int, default=64, help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=1, help="gpu")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size default 256")
    parser.add_argument("--model", type=str, default='RGCN', help="default Model is RGCN")
    parser.add_argument("--num_layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("-e","--num_epochs", type=int, default=4, help="number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout %")
    parser.add_argument("--log_dir", type=str, default='experiment_log/', help="log_dir")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
    parser.add_argument("--bi", type=bool, default=True, help="BiLSTM")
    
    args = parser.parse_args()
    
    main(args)    

