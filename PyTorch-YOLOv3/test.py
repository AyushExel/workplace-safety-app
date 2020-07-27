from __future__ import division
import wandb
from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import artifact_utils

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    class_id_to_label = {
    0: "helmet",
    1: "mask",
    2: "mask"
    }
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    log_imgs=[]

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        '''
        Log the Bounding boxes:
        '''
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            # Output shape => X*X*7
        
        for i,batch_detection in enumerate(outputs):
            bbox_data = []
            if batch_detection is not None:
                bbox_data = [{
                            "position": {
                                "minX": float(img[0]),
                                "maxX": float(img[2]),
                                "minY": float(img[1]),
                                "maxY": float(img[3]),
                            },
                            "class_id" : int(img[6]),
                            "scores" : {
                                "Object_conf": float(img[4]),
                                "class_score": float(img[5])
                            },                          
                            "domain":"pixel"

                        } for img in batch_detection.cpu().numpy()] 
            log_imgs.append(wandb.Image(imgs[i].permute(1, 2, 0).cpu().numpy(), boxes={"predictions": {"box_data":bbox_data , "class_labels": class_id_to_label}}))        

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        wandb.log({"Outputs": log_imgs})

    
    # Concatenate sample statistics
    precision, recall, AP, f1, ap_class = np.array([0]),np.array([0]),np.array([0]),np.array([0]),np.array([0])
    if len(sample_metrics) != 0:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    #parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--model_artifact", type=str, default='demo-model:latest', help="model atrifact to be used to testing")
    parser.add_argument("--job_type", type=str, default='train-eval' , help="job name to uniquely identify the operation")
    parser.add_argument("--name", type=str, default='run' , help="experiment name to uniquely identify the runs")
    opt = parser.parse_args()
    print(opt)
    run = artifact_utils.init_new_run(opt.name,opt.job_type)
    artifact = run.use_artifact(opt.model_artifact, type='model')
    artifact_dir = artifact.download()
    print(artifact_dir)
    artifact_model = os.listdir(artifact_dir)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if artifact_model.endswith(".pth"):
        # Load darknet weights
        model.load_state_dict(torch.load(artifact_dir+'/'+artifact_model))

    else:
        # Load checkpoint weights
        model.load_darknet_weights(opt.weights_path)

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )
    wandb.log({"val_precision": precision.mean(),
                         "val_recall": recall.mean()  ,
                         "val_mAP": AP.mean() ,
                         "val_f1": f1.mean() })
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        wandb.log({"Class "+class_names[c]:AP[i]})
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
