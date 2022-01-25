import pandas as pd
import cv2
import numpy as np
import insightface
import argparse
from sklearn.metrics import roc_curve, accuracy_score

handler = insightface.model_zoo.get_model('models/face_recog.onnx')
handler.prepare(ctx_id=0, providers=['CPUExecutionProvider'])

def Var(x:np.ndarray):
    n = len(x)
    x_hat = x.mean()
    return n * np.sum((x-x_hat)**2)

def NED(u,v, epsilon=1e-8):
    ned = 0.5 * Var(u-v) / (Var(u) + Var(v) + epsilon)
    return 1 - ned**0.5 if args.sqrt else 1 - ned

def get_pairs(pair_path):
    df = pd.read_csv(pair_path)
    return df.values

def save_feat(imgs:list):
    faces_feat = []
    for img in imgs:
        try :
            faces_feat.append(handler.get_feat(cv2.imread(img)))
        except Exception as e:
            print(e)
            print(img)
    return {key:val for key,val in zip(imgs, faces_feat)}

def get_metrics(true_labels, predict_scores_recognize):
    at_thresholds = [0.1, 0.01, 0.001]
    tpr_fpr_result = {}
    for at_threshold in at_thresholds:
        tpr_fpr_result[f"{at_threshold}"] = {
            "TPR": [],
            "FPR": [],
            "final_result": None
        }
        for threshold in np.arange(0, 1+at_threshold, at_threshold):
            predictions = []
            for predict_score_recognize in predict_scores_recognize:
                prediction = predict_score_recognize > threshold
                predictions.append(prediction)
            TP, FP, TN, FN = 0, 0, 0, 0
            for number in range(len(true_labels)):
                if true_labels[number] == 1 and predictions[number] == 1:
                    TP += 1
                elif true_labels[number] == 0 and predictions[number] == 1:
                    FP += 1
                elif true_labels[number] == 0 and predictions[number] == 0:
                    TN += 1
                elif true_labels[number] == 1 and predictions[number] == 0:
                    FN += 1
            TPR = TP/(TP+FN)
            FPR = FP/(FP+TN)
            tpr_fpr_result[f"{at_threshold}"]["TPR"].append(TPR)
            tpr_fpr_result[f"{at_threshold}"]["FPR"].append(FPR)
            
        tpr_fpr_result[f"{at_threshold}"]["TPR"] = sorted(tpr_fpr_result[f"{at_threshold}"]["TPR"])
        tpr_fpr_result[f"{at_threshold}"]["FPR"] = sorted(tpr_fpr_result[f"{at_threshold}"]["FPR"])
        tpr_fpr_result[f"{at_threshold}"]["final_result"] = np.interp(f"{at_threshold}", 
                                                                 tpr_fpr_result[f"{at_threshold}"]["FPR"],
                                                                 tpr_fpr_result[f"{at_threshold}"]["TPR"])
    print(f"TPR@FPR 0.1: {tpr_fpr_result['0.1']['final_result']}%")
    print(f"TPR@FPR 0.01: {tpr_fpr_result['0.01']['final_result']}%")
    print(f"TPR@FPR 0.001: {tpr_fpr_result['0.001']['final_result']}%")
    
    return tpr_fpr_result

def find_th(fpr, tpr, threshold):
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def eval(pairs:np.ndarray, threshold:int=0.6):
    imgs_1 = pairs[:,0].copy()
    imgs_2 = pairs[:,1].copy()
    labels = pairs[:,2].copy().astype(int)

    feats_1 = save_feat(np.unique(imgs_1).tolist())
    feats_2 = save_feat(np.unique(imgs_2).tolist())

    dist = np.array([NED(feats_1[x], feats_2[y]) for x,y in zip(imgs_1, imgs_2)])
    assert len(dist) == len(labels)
    # threshold = np.repeat(threshold, len(dist))
    # pred = np.greater_equal(dist, threshold).astype(int)

    if args.manual:
        _ = get_metrics(labels, dist)
    else:
        fpr, tpr, thresh = roc_curve(labels, dist)
        best_th = find_th(fpr, tpr, thresh)
        # print(tpr)
        # print("==================")
        # print(fpr)
        # print("==================")
        # print(accuracy_score(labels, pred))
        for i in [0.1, 0.01, 0.001]:
            print("TPR@FPR {}: {}".format(i, np.interp(i, fpr, tpr)))
        print("Best threshold : {}".format(best_th))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pair", required=True, type=str)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--sqrt", action="store_true")

    args = parser.parse_args()

    eval(get_pairs(args.pair))