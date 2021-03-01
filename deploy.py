import math
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from model import ST_MTL_SEG
from dataset import SurgicalDataset
from utils import seed_everything, calculate_dice, calculate_confusion_matrix_from_arrays
import warnings
warnings.filterwarnings("ignore")


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
    parser = argparse.ArgumentParser(description='Instrument Segmentation')
    parser.add_argument('--num_classes', default=8, type=int, help="num of classes")
    parser.add_argument('--data_root', default='Instrument_17', help="data root dir")
    parser.add_argument('--batch_size', default=2, type=int, help="num of classes")
    args = parser.parse_args()
    dataset_test = SurgicalDataset(data_root=args.data_root, seq_set=[4,7], is_train=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2,
                              drop_last=True)
    
    print('Sample size of test dataset:', dataset_test.__len__())
    model = ST_MTL_SEG(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load('best_epoch_st-mtl.pth.tar'))
    dices_per_class = validate(test_loader, model, args)
    print('Mean Avg Dice:%.4f [Bipolar Forceps:%.4f, Prograsp Forceps:%.4f, Large Needle Driver:%.4f, Vessel Sealer:%.4f]'
        %(dices_per_class[:4].mean(),dices_per_class[0], dices_per_class[1],dices_per_class[2],dices_per_class[3]))
    
    
if __name__ == '__main__':
    class_names = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Vessel Sealer", "Grasping Retractor", "Monopolar Curve, Scissors", "Other"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything()
    main()