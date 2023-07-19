import pandas as pd
import numpy as np
import os
import torch
import dgl
from torch.optim import Adam
from model import *
from utils import collate_data
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
    
def eval(model, test_dataloader, device):
    # Evaluate AUC, ACC, F1, Prec, Recall
    model.eval()
    val_labels = []
    val_preds = []
    for iter_idx, batch in tqdm(enumerate(test_dataloader, start=1), total=len(test_dataloader)):
        with torch.no_grad():
            one_enroll_graph = batch[0]
            two_enroll_graph = batch[1]
            third_enroll_graph = batch[2]
            labels = batch[3]

            preds = model(one_enroll_graph.to(device), two_enroll_graph.to(device), third_enroll_graph.to(device), False)
            
        val_labels.extend(labels[:,3].cpu().tolist())
        val_preds.extend(preds.cpu().tolist())
    
    val_auc = roc_auc_score(val_labels, val_preds)
    val_acc = accuracy_score(list(map(round, val_labels)), list(map(round, val_preds)))
    val_f1 = f1_score(list(map(round, val_labels)), list(map(round, val_preds)))
    val_precision = precision_score(list(map(round,val_labels)), list(map(round,val_preds)))
    val_recall = recall_score(list(map(round,val_labels)), list(map(round,val_preds)))
    
    return val_auc, val_acc, val_f1, val_precision, val_recall

def train(args, dataloader, logger):
    dgl.random.seed(0)
    torch.manual_seed(0)
    
    if args.dataset == 'kddcup':
        path = 'data/kddcup15'    
        truth_train = pd.read_csv(os.path.join(path, 'truth_train.csv'), header=None, names=['enrollment_id', 'truth'])
        truth_test = pd.read_csv(os.path.join(path, 'truth_test.csv'), header=None, names=['enrollment_id', 'truth'])
            
        train_sampler = SubsetRandomSampler(list(truth_train['enrollment_id']))
        test_sampler = SubsetRandomSampler(list(truth_test['enrollment_id']))
    else:
        path = 'data'    
        X_train = pd.read_pickle(os.path.join(path, 'X_train_df_Naver.pkl'))
        X_test = pd.read_pickle(os.path.join(path, 'X_test_df_Naver.pkl'))
        
        train_sampler = SubsetRandomSampler(list(X_train.index))
        test_sampler = SubsetRandomSampler(list(X_test.index))
        
    device = torch.device("cuda:0" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
        logger.info('gpu is working')
    torch.cuda.empty_cache()
    
    g = dataloader[0]
    input_dim = g.ndata['feature'].shape[1]
    
    #-------------------------------- training Course-------------------------------- #
    logger.info("Loading network finished ...\n")
    logger.info(f"Start training ...\n")
    
    train_dataloader = DataLoader(dataloader, collate_fn=collate_data, sampler=train_sampler ,batch_size = args.batch_size, num_workers = args.num_workers)
    test_dataloader = DataLoader(dataloader, collate_fn=collate_data, sampler=test_sampler, batch_size = args.batch_size, num_workers = args.num_workers)
    
    auc_list, f1_list = [], []
    for i in range(1, 5+1):
        if args.model == 'GraphSage':
            model = Multi_GraphSage(args.num_layers, input_dim, 
                        args.hidden_dim, args.bi, args.dropout, args.batch_size, device).to(device) 
            
        if args.model == 'GAT':
            model = Multi_GAT(args.num_layers, input_dim, 
                        args.hidden_dim, args.bi, args.dropout, args.batch_size, device).to(device) 
            
        if args.model == 'GIN':
            model = Multi_GIN(args.num_layers, input_dim, 
                        args.hidden_dim, args.bi, args.dropout, args.batch_size, device).to(device) 
            
        if args.model == 'RGCN':
            model = Multi_RGCN(args.num_layers, input_dim, 
                        args.hidden_dim, args.bi, args.dropout, args.batch_size, device).to(device) 

        optimizer = Adam(model.parameters(), lr=args.lr)
        mse_loss_fn = nn.BCELoss().to(device)   
        for e in range(1, args.num_epochs+1):
            model.train()
            epoch_loss = 0.
            
            for iter_idx, batch in tqdm(enumerate(train_dataloader, start=1), total=len(train_dataloader)):
                
                one_enroll_graph = batch[0]
                two_enroll_graph = batch[1]
                third_enroll_graph = batch[2]
                labels = batch[3]

                preds = model(one_enroll_graph.to(device), two_enroll_graph.to(device), third_enroll_graph.to(device), True)
                loss = mse_loss_fn(preds.to(device), labels[:,3].to(device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * preds.shape[0]
                
            # eval
            if e == args.num_epochs:
                
                auc, acc, f1, precision, recall = eval(model, test_dataloader, device)

                logger.info(f"**********The {i} times testing AUC is {auc:.4f}********")
                logger.info(f"**********The {i} times testing ACC is {acc:.4f}********")
                logger.info(f"**********The {i} times testing F1 is {f1:.4f}********")
                logger.info(f"**********The {i} times testing Precision is {precision:.4f}********")
                logger.info(f"**********The {i} times testing Recall is {recall:.4f}********")
                logger.info("\n")
        auc_list.append(auc)
        f1_list.append(f1)
        
    auc_mean = np.mean(auc_list)*100
    f1_mean = np.mean(f1_list)*100
    auc_std = np.std(auc_list)*100
    f1_std = np.std(f1_list)*100
    logger.info(f"**********The final testing AUC is {auc_mean:.2f}±{auc_std:.2f}********")
    logger.info(f"**********The final testing F1 is {f1_mean:.2f}±{f1_std:.2f}********")
    logger.info("\n")