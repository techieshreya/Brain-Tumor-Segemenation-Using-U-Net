import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Define image dimensions (should match your training)
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Define your image and mask directories (should match your training)
IMAGE_DIR = r"image_dataset/images/2.png"
MASK_DIR = r"C:\Users\Shreya\Desktop\pr js\image_dataset\masks\2.png"
BATCH_SIZE = 8

# Function to load the dataset (same as in your training script)
def load_dataset(image_dir, mask_dir, img_height, img_width, batch_size, seed=123):
    image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        label_mode=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        seed=seed
    )
    mask_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        mask_dir,
        label_mode=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
        seed=seed
    )
    dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
    dataset = dataset.map(lambda x, y: (x / 255.0, y / 255.0))
    return dataset

# Load your test dataset
test_dataset = load_dataset(IMAGE_DIR, MASK_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, seed=789)

# Custom Dice Coefficient Metric (needed to load models if you used it as a metric)
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + tf.keras.backend.epsilon())

# Function to load a trained model
def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient})

def visualize_segmentation(model, dataset, model_name, save_path, num_examples=1):
    plt.figure(figsize=(15, 5 * num_examples))
    plt.suptitle(f"Predictions from {model_name}", fontsize=16)
    for i, (image_batch, mask_batch) in enumerate(dataset.take(num_examples)):
        predicted_mask_batch = model.predict(image_batch)

        for j in range(min(image_batch.shape[0], 3)):
            # Subplot 1: Input Image
            plt.subplot(num_examples, 3, 1)
            plt.imshow(image_batch[j].numpy())
            plt.title("Input Image")
            plt.axis('off')

            # Subplot 2: True Mask
            plt.subplot(num_examples, 3, 2)
            plt.imshow(mask_batch[j].numpy().squeeze(), cmap='gray')
            plt.title("True Mask")
            plt.axis('off')

            # Subplot 3: Predicted Mask
            plt.subplot(num_examples, 3, 3)
            predicted_mask = (predicted_mask_batch[j].squeeze() > 0.5).astype(np.uint8)
            plt.imshow(predicted_mask, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')
        if i == num_examples - 1:
            break
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # Define the paths to your saved models
    model_paths = {
        'resnet': r'C:\Users\Shreya\Desktop\pr js\unet_resnet_model.h5',
        'mobilenet': r'C:\Users\Shreya\Desktop\pr js\unet_mobilenet_model.h5',
        'vgg': r'C:\Users\Shreya\Desktop\pr js\unet_vgg_model.h5',
        'efficientnet': r'C:\Users\Shreya\Desktop\pr js\unet_efficientnet_model.h5'
    }


    # Create a directory to save the visualization images
    output_dir = "saved_model_predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through your saved models and visualize predictions
    for encoder_type, model_path in model_paths.items():
        print(f"Loading model: {model_path}")
        try:
            loaded_model = load_trained_model(model_path)
            save_path = os.path.join(output_dir, f"unet_{encoder_type}_predictions.png")
            visualize_segmentation(loaded_model, test_dataset.take(1), f"U-Net with {encoder_type} Encoder (Loaded)", save_path, num_examples=1) # Take 1 batch
            print(f"Saved visualization for {encoder_type} encoder to {save_path}")
        except Exception as e:
            print(f"Error loading or visualizing model {encoder_type}: {e}")

    print(f"\nVisualizations saved to the '{output_dir}' directory.")

