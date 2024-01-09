# YOLOv5 precision recall f1 score

# Requirements 
- PyTorch
- NumPy

# Example Run
- Assume you have defined the precision, recall, and F1 score functions within the script
- now call the calculate_precision_recall_f1 as follows

```
precisions, recalls, f1_scores = calculate_precision_recall_f1(predicted_boxes, true_boxes, iou_threshold)
```
- Now run a loop over any of the lists for calculating the metrics as follows

```
for i in range(len(precisions)):
    p`rint(f"Image {i + 1} - Precision: {precisions[i]:.2f}, Recall: {recalls[i]:.2f}, F1 Score: {f1_scores[i]:.2f}")
```
