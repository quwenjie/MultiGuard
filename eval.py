import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
import argparse
from PIL import Image
import numpy as np
from statsmodels.stats.proportion import (
    proportion_confint,
    multinomial_proportions_confint,
)
from scipy.stats import norm
import pickle
from src.models import create_model
from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from voc import *
from nus import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from probability import Get_Overlap
import random
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="ASL MS-COCO Inference on a single image")

parser.add_argument("--model_name", type=str, default="tresnet_xl")
parser.add_argument("--model_path", type=str, default="voc_asl.pth")
parser.add_argument("--input_size", type=int, default=448)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--dataset_type", type=str, default="PASCAL-VOC")

parser.add_argument("--begin", type=float, default=0)
parser.add_argument("--end", type=float, default=2)
parser.add_argument("--T", type=int, default=100)

parser.add_argument("--N", type=int, default=100)
parser.add_argument("--M", type=int, default=500)
parser.add_argument("--sigma", type=float, default=0.1)
parser.add_argument("--alpha", type=float, default=0.001)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--k_prime", type=int, default=1)
parser.add_argument("--record", type=str, default="record.txt")

args = parse_args(parser)


def Randomized_Smoothing(model, x, k, kprime, sigma, iteration, class_num):
    avg = 0
    model.eval()
    for i in range(iteration):
        noise = sigma * torch.randn(x.shape).cuda()
        with torch.no_grad():
            ot = model(torch.clamp(x + noise, 0, 1))
            _, indices = torch.topk(ot, kprime, 1)

            bin = F.one_hot(indices, class_num)
            bin = bin.permute(1, 0, 2)
            dt = 0
            for i in range(kprime):
                dt = dt + bin[i]
            bin = dt
            avg = avg + bin

    _, indices = torch.topk(avg, k, 1)
    ind = indices.permute(1, 0)
    bin = 0
    for i in range(0, k):
        oh = F.one_hot(ind[i], class_num)
        bin = oh + bin

    return indices.cpu().numpy(), bin, avg.cpu().numpy()


def multi_ci(counts, alpha):
    multi_list = []
    n = np.sum(counts)
    l = len(counts)
    for i in range(l):
        multi_list.append(
            proportion_confint(
                min(max(counts[i], 1e-10), n - 1e-10),
                n,
                alpha=alpha * 2.0 / l,
                method="beta",
            )
        )
    return np.array(multi_list)


def binary_search_compute_r(p_sn, p_ls, sigma_value, n):
    p_sn = min(p_sn, 1 - p_ls)
    lower_bound, upper_bound = 0.0, 10.0
    if p_ls <= p_sn / n:
        return 0.0
    while np.abs(lower_bound - upper_bound) > 1e-5:
        searched_radius = (lower_bound + upper_bound) / 2.0
        if (
            norm.cdf(norm.ppf(p_ls) - searched_radius / sigma_value)
            >= norm.cdf(norm.ppf(p_sn) + searched_radius / sigma_value) / n
        ):
            lower_bound = searched_radius
        else:
            upper_bound = searched_radius
    return lower_bound


def CertifyRadius(ls, probability_array, topk, sigma_value):
    p_ls = probability_array[ls]
    probability_array[ls] = -1
    sorted_index = np.argsort(probability_array)[::-1]
    sorted_probability_topk = probability_array[sorted_index[0:topk]]
    p_sk = np.zeros([topk], dtype=np.float)
    radius_array = np.zeros([topk], dtype=np.float)
    for i in np.arange(sorted_probability_topk.shape[0]):
        p_sk[0 : i + 1] += sorted_probability_topk[i]
    for i in np.arange(topk):
        radius_array[i] = binary_search_compute_r(p_sk[i], p_ls, sigma_value, topk - i)
    return np.amax(radius_array)


def get_radius(ls, class_freq):
    CI = multi_ci(class_freq, args.alpha)
    pABar = CI[ls][0]
    probability_bar = CI[:, 1]
    probability_bar = np.clip(probability_bar, a_min=-1, a_max=1 - pABar)
    probability_bar[ls] = pABar
    r = CertifyRadius(ls, probability_bar, args.k, args.sigma)
    return r


def eval_RS(model, data_loader, class_num):
    model.eval()

    # hyperameters
    n = args.N  # randomized smoothing iterations
    sigma = args.sigma  # co-variation of guassian
    alpha = args.alpha  # confidence
    k = args.k  # smoothing output num
    k_prime = args.k_prime  # hard output num

    precision, recall, c_precision, c_recall = 0, 0, 0, 0
    M = 0
    c_num = [0 for i in range(args.T + 10)]
    c_num2 = [0 for i in range(args.T + 10)]
    p_dom = 0
    r_dom = 0
    fi = open(args.record, "a")
    print(
        "N:%d L:%f R:%f k:%d k_prime:%d sigma:%f alpha:%f"
        % (args.N, args.begin, args.end, args.k, args.k_prime, args.sigma, args.alpha),
        file=fi,
    )
    fi.close()  # delete record and rewrite

    for i, (input, target) in enumerate(data_loader):
        # target 1,-1
        if len(target.shape) == 3:
            targ = target.max(dim=1)[0].cuda()
        else:
            targ = F.relu(target).cuda()

        Yi = targ.sum(dim=1)
        with torch.no_grad():
            indices, bin, counts = Randomized_Smoothing(model, input.cuda(), args.k, k_prime, sigma, n, class_num)
            p_dom = p_dom + args.k * targ.shape[0]
            r_dom = r_dom + Yi.sum()
            for i in range(0, args.T + 1, 1):
                perturbation = (args.end - args.begin) / args.T * i + args.begin
                c_overlap = Get_Overlap(counts, targ.cpu(), args.k, k_prime, alpha, perturbation, sigma)
                c_num[i] = c_num[i] + c_overlap.sum()
                c_num2[i] = c_num2[i] + (c_overlap / Yi).sum()

            M += targ.shape[0]
            if M >= args.M:
                break
    fi = open(args.record, "a")
    print("COP:", file=fi)
    for i in range(0, args.T + 1, 1):
        CP_k = (c_num[i] / p_dom).item()
        print("{CP:.4f},".format(CP=CP_k), file=fi)
    print("COR:", file=fi)  # micro recall
    for i in range(0, args.T + 1, 1):
        CR_k = (c_num[i] / r_dom).item()
        print("{CR:.4f},".format(CR=CR_k), file=fi)
    fi.close()
    return 1


def main():
    import os

    availble_gpus = list(range(torch.cuda.device_count()))

    print("Validation code for multi-label classification")

    torch.cuda.empty_cache()
    # setup model
    print("creating and loading the model...")
    state = torch.load(args.model_path, map_location="cpu")
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

    model = create_model(args)

    if (
        args.model_path == "voc_asl.pth"
        or args.model_path == "coco_asl.pth"
        or args.model_path == "nus_asl.pth"
    ):
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_state_dict(state)

    device = torch.device("cuda:0" if len(availble_gpus) > 0 else "cpu")

    model.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    print("data loaded")

    eval_RS(model, val_loader, args.num_classes)


if __name__ == "__main__":
    main()
