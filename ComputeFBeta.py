import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("P")
    parser.add_argument("A")
    parser.add_argument("--iou", default=0.5)
    parser.add_argument("--beta", default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.P) as file:
        preds = json.load(file)

    with open(args.A) as file:
        annos = json.load(file)

    judges = []
    for i, a_value in enumerate(annos):
        iname = a_value["iname"]
        bbox = a_value["bbox"]

        detected = False
        for j, p_value in enumerate(preds):
            if detected:
                break

            if p_value["iname"] == iname:
                iou = compute_iou(p_value["bbox"], bbox)
                if iou > args.iou:
                    detected = True
        judges.append(detected)

    ntp = 0
    for i, j in enumerate(judges):
        if j:
            ntp += 1

    nfn = len(judges) - ntp
    nfp = len(preds) - ntp

    precision = ntp / len(preds)
    recall = ntp / len(judges)
    fbeta = (1 + args.beta ** 2) * precision * recall / ((args.beta ** 2 * precision) + recall)
    return fbeta


def compute_iou(bbox1, bbox2):
    x_min = min(bbox1[0], bbox2[0])
    y_min = min(bbox1[1], bbox2[1])
    x_max = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_max = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    if x_min >= x_max or y_min >= y_max:
        return 0.
    else:
        region = (x_max - x_min) * (y_max - y_min)
        iou = region / (bbox2[2] * bbox2[3])
        return iou


fbeta = main()
print(fbeta)
