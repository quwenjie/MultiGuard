import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.helper_functions.helper_functions import (
    mAP,
    CocoDetection,
    CutoutPIL,
    ModelEma,
    add_weight_decay,
)
import argparse
from PIL import Image
import numpy as np
import pickle
from src.models import create_model
from src.helper_functions.helper_functions import AverageMeter, CocoDetection
from voc import *
from nus import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
from torch.optim import lr_scheduler
import torch.nn.functional as F
import os

parser = argparse.ArgumentParser(description="ASL MS-COCO Inference on a single image")

parser.add_argument("--input_size", type=int, default=448)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--dataset_type", type=str, default="PASCAL-VOC")
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--weightdecay", type=float, default=0)
parser.add_argument("--step", type=int, default=5)  # 40
parser.add_argument("--convepoch", type=int, default=50)  # 40
parser.add_argument("--epoch", type=int, default=100)  # 100
parser.add_argument("--gamma", type=float, default=0.5)  # 0.1
parser.add_argument("--sigma", type=float, default=1)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--load_w", type=int, default=1)

parser.add_argument("--model_name", type=str, default="tresnet_xl")
parser.add_argument("--log", type=str, default="train.txt")

args = parse_args(parser)


def eval(model, data_loader, sigma):

    model.eval()
    c_num = 0
    sigmoid = torch.nn.Sigmoid()
    p_dom, r_dom, M = 0, 0, 0
    for i, (input, target) in enumerate(data_loader):
        # target 1,-1
        if len(target.shape) == 3:
            targ = target.max(dim=1)[0].cuda()
        else:
            targ = F.relu(target).cuda()
        Yi = targ.sum(dim=1)
        with torch.no_grad():
            noise = sigma * torch.randn(input.shape)
            input = torch.clamp(input + noise, 0, 1)
            # hard classifier: => >0 ,scorer sigmoid
            output = model(input.cuda())

            output = sigmoid(output)
            bin = output.clone()
            bin[output > 0.8] = 1  # threshold
            bin[output <= 0.8] = 0
            p_dom = p_dom + bin.sum()
            r_dom = r_dom + Yi.sum()

            c_num = c_num + (bin == targ).sum()

            M += targ.shape[0]
            if M >= 500:
                break

    OP_k, OR_k = (c_num / p_dom).item(), (c_num / r_dom).item()
    fi = open(args.log, "a")
    print("OP_3: {OP:.4f}\t OR_3: {OR:.4f}\t".format(OP=OP_k, OR=OR_k), file=fi)
    fi.close()
    return 1


def train(model, train_loader, optimizer, scheduler, sigma):

    loss_func = AsymmetricLossOptimized()
    model.train()
    tot = 0
    for i, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # target 1,-1
        if len(target.shape) == 3:
            target = target.max(dim=1)[0].cuda()
        else:
            target = F.relu(target).cuda()
        noise = sigma * torch.randn(input.shape)
        input = torch.clamp(input + noise, 0, 1)
        output = model(input.cuda())
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        tot += loss.item()

    scheduler.step()
    fi = open(args.log, "a")
    print("Train Loss tot: ", tot, file=fi)
    fi.close()

    return 1


def main():

    availble_gpus = list(range(torch.cuda.device_count()))

    print("Validation code for multi-label classification")

    torch.cuda.empty_cache()
    # setup model
    print("creating and loading the model...")

    if args.dataset_type == "PASCAL-VOC":  # ML-GCN
        args.num_classes = 20
        MEAN = [0, 0, 0]
        STD = [1, 1, 1]
        train_dataset = get_voc2007_train(args.input_size, MEAN, STD)
        val_dataset = get_voc2007_test(args.input_size, MEAN, STD)
    elif args.dataset_type == "MS-COCO":
        args.num_classes = 80
        args.do_bottleneck_head = False
        normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        instances_path_val = os.path.join(
            "./COCO", "annotations/instances_val2014.json"
        )
        val_data_path = os.path.join("./COCO", "val2014")
        val_dataset = CocoDetection(
            val_data_path,
            instances_path_val,
            transforms.Compose(
                [
                    transforms.Resize((args.input_size, args.input_size)),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        instances_path_train = os.path.join(
            "./COCO", "annotations/instances_train2014.json"
        )
        train_data_path = os.path.join("./COCO", "train2014")
        train_dataset = CocoDetection(
            train_data_path,
            instances_path_train,
            transforms.Compose(
                [
                    transforms.Resize((args.input_size, args.input_size)),
                    CutoutPIL(cutout_factor=0.5),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
    elif args.dataset_type == "NUS-WIDE":
        args.num_classes = 81
        MEAN = [0, 0, 0]
        STD = [1, 1, 1]
        train_dataset = get_nuswide_train(args.input_size, MEAN, STD)
        val_dataset = get_nuswide_test(args.input_size, MEAN, STD)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=5,
        pin_memory=True,
    )

    model = create_model(args)

    fi = open(args.log, "w")
    fi.close()

    if args.load_w:
        if args.model_name == "tresnet_xl":
            args.model_path = "tresnet_xl_448.pth"  # load tresnet weight as initial
            state = torch.load(args.model_path, map_location="cpu")
            state["model"]["head.fc.bias"] = model.head.fc.bias.clone()
            state["model"]["head.fc.weight"] = model.head.fc.weight.clone()
            model.load_state_dict(state["model"], strict=True)
        elif args.model_name == "tresnet_l":
            args.model_path = "tresnet_l_448.pth"  # load tresnet weight as initial
            state = torch.load(args.model_path, map_location="cpu")
            state["model"]["head.fc.bias"] = model.head.fc.bias.clone()
            state["model"]["head.fc.weight"] = model.head.fc.weight.clone()
            model.load_state_dict(state["model"], strict=True)

    device = torch.device("cuda:0" if len(availble_gpus) > 0 else "cpu")

    model.to(device)

    lr = args.lr
    parameters = add_weight_decay(model, args.weightdecay)
    # true wd, filter_bias_and_bn
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=args.epoch
    )

    for epoch in range(args.epoch):
        t = max(0, args.convepoch - epoch)
        if args.convepoch > 0:
            sigma = (1 - 1 / (args.convepoch * args.convepoch) * t * t) * args.sigma
        else:
            sigma = args.sigma
        train(model, train_loader, optimizer, scheduler, sigma)
        eval(model, val_loader, args.sigma)
        torch.save(model.state_dict(), "save_%f_%d.pth" % (args.lr, epoch))


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True
    main()
