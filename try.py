import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define image dimensions
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Paths to your single test image and corresponding mask
IMAGE_PATH = r"image_dataset/images/2.png"
MASK_PATH = r"C:\Users\Shreya\Desktop\pr js\image_dataset\masks\2.png"

# Custom Dice Coefficient Metric
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + tf.keras.backend.epsilon())

# Load and preprocess a single image and mask
def load_single_image_and_mask(image_path, mask_path):
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    mask = load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
    mask = img_to_array(mask) / 255.0
    mask = np.expand_dims(mask, axis=0)  # Add batch dimension

    return image, mask

# Visualize single prediction
def visualize_prediction(model, image, mask, model_name, save_path):
    prediction = model.predict(image)
    predicted_mask = (prediction[0].squeeze() > 0.5).astype(np.uint8)

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Prediction from {model_name}", fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(image[0])
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask[0].squeeze(), cmap='gray')
    plt.title("True Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Main
if __name__ == '__main__':
    model_paths = {
        'resnet': r'C:\Users\Shreya\Desktop\pr js\unet_resnet_model.h5',
        'mobilenet': r'C:\Users\Shreya\Desktop\pr js\unet_mobilenet_model.h5',
        'vgg': r'C:\Users\Shreya\Desktop\pr js\unet_vgg_model.h5',
        'efficientnet': r'C:\Users\Shreya\Desktop\pr js\unet_efficientnet_model.h5'
    }

    os.makedirs("model_predictions", exist_ok=True)

    # Load one test image and mask
    image, mask = load_single_image_and_mask(IMAGE_PATH, MASK_PATH)

    for encoder_type, model_path in model_paths.items():
        print(f"Loading model: {model_path}")
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient})
            save_path = f"model_predictions/unet_{encoder_type}_prediction.png"
            visualize_prediction(model, image, mask, f"U-Net with {encoder_type} Encoder", save_path)
            print(f"Saved prediction for {encoder_type} to {save_path}")
        except Exception as e:
            print(f"Error loading model {encoder_type}: {e}")

    print("\nAll predictions saved to 'saved_model_predictions'.")
