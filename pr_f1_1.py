import torch

# Example data for predicted and true boxes as PyTorch tensors
predicted_boxes = torch.tensor([
    [236.9800, 142.5100, 261.6800, 212.0100],
    [7.0300, 167.7600, 156.3500, 262.6300],
    [557.2100, 209.1900, 638.5600, 287.9200],
    [358.9800, 218.0500, 414.9800, 320.8800],
    [290.6900, 218.0000, 352.5200, 316.4800],
    [413.2000, 223.0100, 443.3700, 304.3700],
    [317.4000, 219.2400, 338.9800, 230.8300],
    [412.8000, 157.6100, 465.8500, 295.6200]
], device='cuda:0')

true_boxes = torch.tensor([
    [236.9800, 142.5100, 261.6800, 212.0100],
    [7.0300, 167.7600, 156.3500, 262.6300],
    [557.2100, 209.1900, 638.5600, 287.9200],
    [358.9800, 218.0500, 414.9800, 320.8800],
    [290.6900, 218.0000, 352.5200, 316.4800],
    [413.2000, 223.0100, 443.3700, 304.3700],
    [317.4000, 219.2400, 338.9800, 230.8300],
    [412.8000, 157.6100, 465.8500, 295.6200]
], device='cuda:0')

iou_threshold = 0.5  # IoU threshold for matching predictions to ground truth

def calculate_iou_p(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xA = torch.max(x1, x1_)
    yA = torch.max(y1, y1_)
    xB = torch.min(x2, x2_)
    yB = torch.min(y2, y2_)
    inter_area = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def calculate_precision_recall_f1(predicted_boxes, true_boxes, iou_threshold):
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(predicted_boxes.size(0)):
        true_positives = 0
        false_positives = 0
        total_true_boxes = true_boxes.size(0)
        total_predicted_boxes = predicted_boxes.size(0)

        if total_true_boxes == 0 or total_predicted_boxes == 0:
            # Handle cases where there are no true boxes or predicted boxes
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
        else:
            for j in range(total_predicted_boxes):
                matched = False
                for k in range(total_true_boxes):
                    iou = calculate_iou(predicted_boxes[j], true_boxes[k])
                    if torch.max(iou) >= iou_threshold:
                        true_positives += 1
                        matched = True
                        break

                if not matched:
                    false_positives += 1

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / total_true_boxes
            f1_score = 2 * (precision * recall) / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return precisions, recalls, f1_scores

# Calculate precision, recall, and F1 score per image
precisions, recalls, f1_scores = calculate_precision_recall_f1(predicted_boxes, true_boxes, iou_threshold)

# Now, you have precision, recall, and F1 score values per image
for i in range(len(precisions)):
    print(f"Image {i + 1} - Precision: {precisions[i]:.2f}, Recall: {recalls[i]:.2f}, F1 Score: {f1_scores[i]:.2f}")
