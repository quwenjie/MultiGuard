# MultiGuard

Official code for paper MultiGuard: Provably Robust Multi-label Classification against Adversarial Examples, accepted by Neurips 2022.

Pretrained robust models will be released soon.

## Dataset Preparation

Please place PASCAL-VOC in `./voc2007` , MS-COCO in `COCO`, NUS-WIDE in `./NUS_WIDE`.

## Evaluation

Place pretrained models under `./models`

Evaluate perturbation range from [0,3], on VOC2007 dataset, using $k=5,k'=1,\sigma=0.5,\alpha=10^{-3}$, test on 500 images. 

```bash
python3 eval.py --begin=0 --end=3 --T=300 --N=1000  --sigma=0.5 --batch_size=32 --k=5 --k_prime=1 --alpha=0.001 --record=test_voc_result.txt --dataset_type=PASCAL-VOC --model_name=tresnet_xl --model_path=./models/voc_asl_0.5.pth --M=500
```

Evaluate on MS-COCO dataset, using $k=10,k'=2,\sigma=0.5$. 

```bash
python3 eval.py --begin=0 --end=3 --T=300 --N=1000  --sigma=0.5 --batch_size=32 --k=10 --k_prime=2 --alpha=0.001 --record=test_coco_result.txt --dataset_type=MS-COCO --model_name=tresnet_l --model_path=./models/coco_asl_0.5.pth --M=500
```

Evaluate on NUS-WIDE dataset, using $k=10,k'=2,\sigma=0.5$. 

```bash
python3 eval.py --begin=0 --end=3 --T=300 --N=1000  --sigma=0.5 --batch_size=32 --k=10 --k_prime=2 --alpha=0.001 --record=test_nus_result.txt --dataset_type=NUS-WIDE --model_name=tresnet_l --model_path=./models/nus_asl_0.5.pth --M=500
```

## Train

Train randomized smoothing model on VOC2007 dataset with $\sigma=0.5$.

```bash
python3 train.py --batch_size=16 --dataset_type=PASCAL-VOC --model_name=tresnet_xl --sigma=0.5 --log=train_voc.txt
```
