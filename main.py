import math
import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import ST_MTL_SEG
from dataset import SurgicalDataset
from utils import seed_everything, calculate_dice, calculate_confusion_matrix_from_arrays

def train(train_loader, model, criterion, optimizer, epoch, epoch_iters):
    model.train()
    for batch_idx, (inputs, labels_seg, _) in enumerate(train_loader):
        inputs, labels_seg = inputs.to(device), labels_seg.to(device)
        optimizer.zero_grad()
        pred_seg = model(inputs)
        loss = criterion(pred_seg, labels_seg)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.4f]' % (
                epoch, batch_idx + 1, epoch_iters, loss.item(),
                optimizer.param_groups[0]['lr']))

def validate(valid_loader, model, args):
    confusion_matrix = np.zeros(
            (args.num_classes, args.num_classes), dtype=np.uint32)
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels_seg, _,_) in enumerate(valid_loader):
            inputs, labels_seg = inputs.to(device), np.array(labels_seg)
            pred_seg = model(inputs)
            pred_seg = pred_seg.data.max(1)[1].squeeze_(1).cpu().numpy()
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                pred_seg, labels_seg, args.num_classes)    

    confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
    dices = {'dice_{}'.format(cls + 1): dice
                for cls, dice in enumerate(calculate_dice(confusion_matrix))}
    dices_per_class = np.array(list(dices.values()))          

    return dices_per_class

def main():
    dataset_train = SurgicalDataset(data_root=args.data_root, seq_set = [1,2,3,5,6,8], is_train=True)
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)

    dataset_test = SurgicalDataset(data_root=args.data_root, seq_set=[4,7], is_train=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2,
                              drop_last=True)
    model = ST_MTL_SEG(num_classes=args.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print('Length of dataset- train:', dataset_train.__len__(), ' valid:', test_loader.__len__())
    epoch_iters = dataset_train.__len__() / args.batch_size
    best_dice = 0
    best_epoch = 0
    for epoch in range( args.num_epoch):
        train(train_loader, model, criterion, optimizer, epoch, epoch_iters)
        dices_per_class = validate(test_loader, model, args)
        avg_dice = dices_per_class[:4].mean() # First 4 classes contains in the valid set
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best_epoch_st-mtl.pth.tar'))

        print('Epoch:%d ' % epoch,'Mean Avg Dice:%.4f [Bipolar F.:%.4f, Prograsp F.:%.4f, Large Needle D.s:%.4f, Vessel Sealer:%.4f]'
        %(dices_per_class[:4].mean(),dices_per_class[0], dices_per_class[1],dices_per_class[2],dices_per_class[3]), 'Best Avg Dice = %d : %.4f ' % (best_epoch, best_dice))
    
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything()
    parser = argparse.ArgumentParser(description='Instrument Segmentation')
    parser.add_argument('--num_classes', default=8, type=int, help="num of classes")
    parser.add_argument('--num_epoch', default=100, type=int, help="num of epochs")
    parser.add_argument('--log_interval', default=100, type=int, help="log interval")
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--data_root', default='Instrument_17', help="data root dir")
    parser.add_argument('--ckpt_dir', default='ckpt', help="data root dir")
    parser.add_argument('--batch_size', default=2, type=int, help="num of classes")
    args = parser.parse_args()
    main()