# Code of the training and validation of SITSMamba
import sys
sys.path.append('/data/xiaolei.qin/Project/SITSMamba/MTLCC-pytorch-master/src')
sys.path.append('/data/xiaolei.qin/Project/SITSMamba/utae-paps-main/utae-paps-main')
import torch.nn

from utils.dataset import ijgiDataset as Dataset
from Models.SITSMamba import UMamba
from utils.logger import  Printer
import argparse
from utils.snapshot import save, resume

from sklearn.metrics import accuracy_score

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', "--datadir", default = '/data/xiaolei.qin/Dataset/MTLCC/data',type=str, help="path to dataset")
    parser.add_argument('-b', "--batchsize", default=4 , type=int, help="batch size")
    parser.add_argument('-w', "--workers", default=4, type=int, help="number of dataset worker threads")
    parser.add_argument('-e', "--epochs", default=100, type=int, help="epochs to train")
    parser.add_argument('-l', "--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument('-s', "--snapshot", default="/data/xiaolei.qin/Project/SITSMamba/Checkpoints/sitsmamba/model.pth", type=str, help="load weights from snapshot")
    parser.add_argument('-c', "--checkpoint_dir", default="/data/xiaolei.qin/Project/SITSMamba/Checkpoints/sitsmamba", type=str, help="directory to save checkpoints")
    return parser.parse_args()

def main(
    datadir,
    batchsize = 16,
    workers = 8,
    epochs = 100,
    lr = 1e-3,
    snapshot = None,
    checkpoint_dir = None
    ):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_seed(3407)

    # # ---------------mtlcc dataset-----------------------------
    traindataset = Dataset(datadir, tileids="tileids/train_fold0.tileids")
    valdataset = Dataset(datadir, tileids="tileids/test_fold0.tileids")
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True, num_workers=workers,
                                                  pin_memory=True)
    valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=batchsize, shuffle=False, num_workers=workers,
                                                pin_memory=True)


    network = UMamba(input_dim=13,
                   encoder_widths=[128],
                   decoder_widths=[128],
                   out_conv=[32, 18],
                   encoder_norm="group",
                   return_maps=False,
                   pad_value=0,
                   padding_mode="reflect",d_state=16)


    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network,device_ids=[0]).cuda()
        loss = loss.cuda()

    start_epoch = 0

    if snapshot is not None:
        state = resume(snapshot,model=network, optimizer=optimizer)

        if "epoch" in state.keys():
            start_epoch = state["epoch"]


    max_acc = 0
    for epoch in range(start_epoch, epochs):
        print("\nEpoch {}".format(epoch))
        print("train")
        Loss_list = 0
        Loss_list1= train_epoch(traindataloader, network, optimizer, loss, Loss_list)
        print('loss,',Loss_list1)

        print("\ntest")
        Acc=val_epoch(valdataloader, network)
        print('Acc',Acc)

        if checkpoint_dir is not None:
            checkpoint_name = os.path.join(checkpoint_dir, "model.pth")
            if Acc > max_acc:
                save(checkpoint_name, network, optimizer, epoch=epoch)
                max_acc = Acc
                print('save weight to', checkpoint_dir)

def train_epoch(dataloader, network, optimizer, loss,Loss_list):
    network.train()
    printer = Printer(N=len(dataloader))

    for iteration, data in enumerate(dataloader):
        optimizer.zero_grad()

        input, target = data

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        ##baseline (w/o RBranch):
        # output = network.forward(input)
        # l = loss(output, target)

        # w/ RBranch:
        output, rec_ts = network.forward(input,True)#单向
        mseloss=torch.nn.MSELoss(reduction='none')
        rec_ts=rec_ts.permute(0,3,4,1,2).contiguous().view(-1,rec_ts.shape[1],rec_ts.shape[2])
        input=input.permute(0,3,4,1,2).contiguous().view(-1,input.shape[1],input.shape[2])

        #PW
        l_rec=(torch.range(1,input.shape[1])/input.shape[1]).unsqueeze(0).unsqueeze(2).repeat(input.shape[0],1,input.shape[2]).cuda()
        l_rec*=mseloss(rec_ts,input)
        l_rec=l_rec.mean()
        l = loss(output, target) + 0.03 * l_rec / (l_rec / loss(output, target)).detach()

        stats = {"loss":l.data.cpu().numpy()}

        l.backward()
        optimizer.step()

        printer.print(stats, iteration)

        Loss_list += l.item()
    return Loss_list
def val_epoch(dataloader, network):
    printer = Printer(N=len(dataloader))
    network.eval()
    firstIter=True
    with torch.no_grad():
        for iteration, data in enumerate(dataloader):

            input, target = data


            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # baseline
            output = network.forward(input)

            output = output.argmax(dim=1)

            if firstIter:
                y_true = target.cpu().detach().numpy()
                y_pred = output.cpu().detach().numpy()
                firstIter = False
            else:
                y_true = np.concatenate((target.cpu().detach().numpy(), y_true), 0)
                y_pred = np.concatenate((output.cpu().detach().numpy(), y_pred), 0)

            printer.print(None, iteration)
        acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    return acc


if __name__ == "__main__":

    args = parse_args()

    main(
        datadir = args.datadir,
        batchsize=args.batchsize,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.learning_rate,
        snapshot=args.snapshot,
        checkpoint_dir=args.checkpoint_dir
    )
