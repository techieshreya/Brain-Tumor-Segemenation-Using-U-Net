import matplotlib.pyplot as plt
import numpy as np
import os

# Create the output directory if it doesn't exist
output_dir = "saved_graph_evaluation"
os.makedirs(output_dir, exist_ok=True)

# Model names
models = ['ResNet', 'MobileNet', 'VGG', 'EfficientNet']

# Your performance metrics (replace with your actual values)
accuracy = np.array([0.997961402, 0.997095942, 0.99714458, 0.982659101])
dice_coefficient = np.array([0.9378286, 0.903375506, 0.907346189, 0.920756519])
precision = np.array([0.974892437, 0.932546854, 0.948966086, 0.93859762])
recall = np.array([0.94106555, 0.941932201, 0.918283463, 0.960916102])
mean_iou = np.array([0.49774617, 0.508197486, 0.521264672, 0.491840571])

# --- 1. Bar Chart for Accuracy ---
plt.figure(figsize=(8, 6))
plt.bar(models, accuracy, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of U-Net Models')
plt.ylim(0.98, 1.0)  # Adjust y-axis limits for better visualization
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
plt.close()

# --- 2. Bar Chart for Dice Coefficient ---
plt.figure(figsize=(8, 6))
plt.bar(models, dice_coefficient, color='lightcoral')
plt.xlabel('Model')
plt.ylabel('Dice Coefficient')
plt.title('Dice Coefficient Comparison of U-Net Models')
plt.ylim(0.90, 0.95)  # Adjust y-axis limits
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dice_coefficient_comparison.png'))
plt.close()

# --- 3. Bar Chart for Precision ---
plt.figure(figsize=(8, 6))
plt.bar(models, precision, color='lightgreen')
plt.xlabel('Model')
plt.ylabel('Precision')
plt.title('Precision Comparison of U-Net Models')
plt.ylim(0.93, 0.98)  # Adjust y-axis limits
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'precision_comparison.png'))
plt.close()

# --- 4. Bar Chart for Recall ---
plt.figure(figsize=(8, 6))
plt.bar(models, recall, color='gold')
plt.xlabel('Model')
plt.ylabel('Recall')
plt.title('Recall Comparison of U-Net Models')
plt.ylim(0.91, 0.97)  # Adjust y-axis limits
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'recall_comparison.png'))
plt.close()

# --- 5. Bar Chart for Mean IoU ---
plt.figure(figsize=(8, 6))
plt.bar(models, mean_iou, color='lightsalmon')
plt.xlabel('Model')
plt.ylabel('Mean IoU')
plt.title('Mean IoU Comparison of U-Net Models')
plt.ylim(0.49, 0.53)  # Adjust y-axis limits
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mean_iou_comparison.png'))
plt.close()

print(f"Bar graphs saved to the '{output_dir}' directory.")