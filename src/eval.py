import torch
from torchmetrics import JaccardIndex
import numpy as np

from backboned_unet import Unet
from datasets import preprocess_eval_data, load_eval_data

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eval_rasters_path = '../data/eval/Rasters/Harmonized/'
labels_path = '../data/eval/Labels/Harmonized/'
model_load_path = '../weights/model_weights_harmonized.pt'

def plot_predictions():
    pass

def get_accuracy_metrics(tensor1, tensor2):
    valid_data_mask = (tensor2 != -1)
    boolean_tensor = torch.eq(tensor1[valid_data_mask].to('cpu'), tensor2[valid_data_mask].to('cpu'))
    return torch.count_nonzero(boolean_tensor), torch.numel(boolean_tensor)

def display_overall_metrics(model, dataloader):
    """
    This function computes metrics such as accuracy, IoU, recall, precision, F1-Score, Balanced Accuracy
    by aggregating calculations across all patches in the dataloader, to evaluate for the inputs as a whole.
    """
    model.eval()
    jaccard = JaccardIndex(task='multiclass', num_classes=2, average = 'micro', ignore_index=-1)
    target_true = 0
    predicted_true, predicted_false = 0, 0
    correct_true, correct_false, wrong_true, wrong_false = 0, 0, 0, 0
    total_match, total_pixels = 0, 0
    for item in dataloader:
        sample = preprocess_eval_data(item)
        X = sample['image']
        mask = sample['mask']
        X = X.to(device)
        y = model(X)
        pred = torch.argmax(y, 1)
        pred_tensor = torch.squeeze(pred, 0)
        pred_tensor = pred_tensor.to('cpu')
        mask_squeezed = torch.squeeze(mask)
        valid_data_mask = (mask_squeezed != -1)
        
        target_true += torch.sum(mask_squeezed == 1).float()
        valid_preds = torch.tensor(np.where(valid_data_mask == 1, pred_tensor, -1))
        predicted_true += torch.sum(valid_preds == 1).float()
        predicted_false += torch.sum(valid_preds == 0).float()
        stage_tensor = valid_preds[valid_preds == mask_squeezed]
        correct_true += torch.sum(stage_tensor == 1).float()
        correct_false += torch.sum(stage_tensor == 0).float()

        jaccard.update(pred_tensor.to('cpu'), mask_squeezed.to('cpu'))
        num_match, num_pixels = get_accuracy_metrics(pred_tensor, mask_squeezed)
        total_match += num_match
        total_pixels += num_pixels

    wrong_true = predicted_true - correct_true
    wrong_false = predicted_false - correct_false
    
    recall_positive = correct_true / target_true # same as sensitivity
    precision_positive = correct_true / predicted_true
    f1_score_positive = 2 * precision_positive * recall_positive / (precision_positive + recall_positive)
    
    recall_negative = correct_false / (correct_false + wrong_true) # same as specificity
    precision_negative = correct_false / predicted_false
    f1_score_negative = 2 * precision_negative * recall_negative / (precision_negative + recall_negative)

    overall_accuracy = total_match / total_pixels
    overall_iou = jaccard.compute()
    bal_acc = (recall_positive + recall_negative) / 2
    
    print("Overall Accuracy Value is " + str(overall_accuracy.item()))
    print("Overall IOU Value is " + str(overall_iou.item()))
    print("Balanced accuracy is " + str(bal_acc.item()))
    print("---------")
    print("Precision for Snow is " + str(precision_positive.item()))
    print("Recall for Snow is " + str(recall_positive.item()))
    print("F1 score for Snow is " + str(f1_score_positive.item()))
    print("---------")
    print("Precision for Non-Snow is " + str(precision_negative.item()))
    print("Recall for Non-Snow is " + str(recall_negative.item()))
    print("F1 score for Non-Snow is " + str(f1_score_negative.item()))
    

def main():
    print(f'Device: {device}')
    model = Unet(backbone_name='resnet50', pretrained=True, encoder_freeze=True, classes=2)
    model = model.to(device)
    model.load_state_dict(torch.load(model_load_path))

    dataloader, num_samples = load_eval_data(eval_rasters_path, labels_path)

    display_overall_metrics(model, dataloader)
    

if __name__ == '__main__':
    main()

