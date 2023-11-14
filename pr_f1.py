import torch

# Example data for predicted and true boxes as PyTorch tensors
predicted_boxes = [
    torch.tensor([[100, 100, 200, 200], [150, 150, 220, 220]]),
    torch.tensor([[280, 280, 410, 410], [220, 220, 300, 300]]),
    torch.tensor([[95, 95, 210, 210], [220, 220, 300, 300]]),
]

true_boxes = [
    torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
    torch.tensor([[280, 280, 410, 410], [95, 95, 210, 210]]),
    torch.tensor([[150, 150, 220, 220], [220, 220, 300, 300]]),
]

iou_threshold = 0.5  # IoU threshold for matching predictions to ground truth

def calculate_iou(box1, box2):
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

    for i in range(len(predicted_boxes)):
        true_positives = 0
        false_positives = 0
        total_true_boxes = true_boxes[i].size(0)
        total_predicted_boxes = predicted_boxes[i].size(0)

        if total_true_boxes == 0 or total_predicted_boxes == 0:
            # Handle cases where there are no true boxes or predicted boxes
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
        else:
            for j in range(total_predicted_boxes):
                matched = False
                for k in range(total_true_boxes):
                    iou = calculate_iou(predicted_boxes[i][j], true_boxes[i][k])
                    if torch.max(iou) >= iou_threshold:
                        true_positives += 1
                        matched = True
                        break

                if not matched:
                    false_positives += 1

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / total_true_boxes
            f1_score = 2 * (precision * recall) / (precision + recall)

        precisions.append(precision.item())
        recalls.append(recall.item())
        f1_scores.append(f1_score.item())

    return precisions, recalls, f1_scores

# Calculate precision, recall, and F1 score per image
precisions, recalls, f1_scores = calculate_precision_recall_f1(predicted_boxes, true_boxes, iou_threshold)

# Now, you have precision, recall, and F1 score values per image
for i in range(len(precisions)):
    print(f"Image {i + 1} - Precision: {precisions[i]:.2f}, Recall: {recalls[i]:.2f}, F1 Score: {f1_scores[i]:.2f}")
