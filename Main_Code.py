import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os  

IMAGE_DIR = r"D:\DL\image_dataset\images"
MASK_DIR = r"D:\DL\image_dataset\masks"
BATCH_SIZE = 8
IMG_HEIGHT, IMG_WIDTH = 128, 128
EPOCHS = 20

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

# Create train, validation, and test datasets
train_dataset = load_dataset(IMAGE_DIR, MASK_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, seed=123)
val_dataset = load_dataset(IMAGE_DIR, MASK_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, seed=456)
test_dataset = load_dataset(IMAGE_DIR, MASK_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, seed=789)

# Define U-Net encoder selection
def build_encoder(input_shape, encoder_type):
    if encoder_type == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        skip_connections = [base_model.get_layer(name).output for name in [
            "conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]]
    elif encoder_type == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        skip_connections = [base_model.get_layer(name).output for name in [
            "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu"]]
    elif encoder_type == 'vgg':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        skip_connections = [base_model.get_layer(name).output for name in [
            "block1_pool", "block2_pool", "block3_pool", "block4_pool"]]
    elif encoder_type == 'efficientnet':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        skip_connections = [base_model.get_layer(name).output for name in [
            "block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation"]]

    return Model(inputs=base_model.input, outputs=skip_connections)

# Decoder part
def build_decoder(encoder_outputs):
    block1, block2, block3, block4 = encoder_outputs
    bottleneck = Conv2D(512, (3, 3), activation='relu', padding='same')(block4)

    up3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    up3 = concatenate([up3, block3])
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up3)
    up2 = concatenate([up2, block2])
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)

    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2)
    up1 = concatenate([up1, block1])
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)

    outputs = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same')(up1)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(outputs)
    return outputs

# Main function to build U-Net model with specific encoder
def build_unet(input_shape, encoder_type):
    inputs = Input(input_shape)
    encoder = build_encoder(input_shape, encoder_type)
    encoder_outputs = encoder(inputs)
    outputs = build_decoder(encoder_outputs)
    return Model(inputs=inputs, outputs=outputs)

# Custom Dice Coefficient Metric
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + tf.keras.backend.epsilon())

# Function to visualize the image, true mask, and predicted mask
def visualize_segmentation(model, dataset, model_name, save_path, num_examples=3):
    plt.figure(figsize=(15, 5 * num_examples))
    plt.suptitle(f"Predictions from {model_name}", fontsize=16)
    for i, (image_batch, mask_batch) in enumerate(dataset.take(num_examples)):
        predicted_mask_batch = model.predict(image_batch)

        for j in range(image_batch.shape[0]):
            plt.subplot(num_examples, 3, i * 3 + j + 1)
            plt.imshow(image_batch[j].numpy())
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(num_examples, 3, i * 3 + j + 2)
            plt.imshow(mask_batch[j].numpy().squeeze(), cmap='gray')
            plt.title("True Mask")
            plt.axis('off')

            plt.subplot(num_examples, 3, i * 3 + j + 3)
            predicted_mask = (predicted_mask_batch[j].squeeze() > 0.5).astype(np.uint8)
            plt.imshow(predicted_mask, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')
        if i == num_examples - 1:
            break
    plt.tight_layout()
    plt.savefig(save_path)  # Save the plot
    plt.close() # Close the plot to free up memory

# Metrics dictionary to store results
metrics_results = {'Model': [], 'Accuracy': [], 'Dice Coefficient': [], 'Precision': [], 'Recall': [], 'Mean IoU': []}

# Define the models to compare
models = ['resnet', 'mobilenet', 'vgg', 'efficientnet']
input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

# Create a directory to save the visualization images
os.makedirs("segmentation_predictions", exist_ok=True)

# Train and evaluate each model
for encoder_type in models:
    print(f"\nTraining U-Net with {encoder_type} encoder...")

    model = build_unet(input_shape=input_shape, encoder_type=encoder_type)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', dice_coefficient, tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(), tf.keras.metrics.MeanIoU(num_classes=2)])

    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, verbose=2)

    test_loss, test_accuracy, test_dice, test_precision, test_recall, test_mean_iou = model.evaluate(test_dataset)

    # Save results
    metrics_results['Model'].append(encoder_type)
    metrics_results['Accuracy'].append(test_accuracy)
    metrics_results['Dice Coefficient'].append(test_dice)
    metrics_results['Precision'].append(test_precision)
    metrics_results['Recall'].append(test_recall)
    metrics_results['Mean IoU'].append(test_mean_iou)

    # Save model
    model.save(f'unet_{encoder_type}_model.h5')
    print(f"Trained and saved U-Net with {encoder_type} encoder.")

    # Visualize predictions for the current model on the test dataset and save the plot
    save_path = f"segmentation_predictions/unet_{encoder_type}_predictions.png"
    visualize_segmentation(model, test_dataset.take(2), f"U-Net with {encoder_type} Encoder", save_path, num_examples=3)
    print(f"Saved visualization for {encoder_type} encoder to {save_path}")

# Save the metrics to an Excel file
df = pd.DataFrame(metrics_results)
df.to_excel("model_comparison_results.xlsx", index=False)
print("\nResults saved to model_comparison_results.xlsx")