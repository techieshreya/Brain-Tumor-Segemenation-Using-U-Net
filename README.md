## Brain Tumor Segmentation using U-Net (TensorFlow/Keras)

This repository trains and compares U-Net segmentation models with different ImageNet-pretrained encoders — ResNet50, MobileNetV2, VGG16, and EfficientNetB0 — for brain tumor segmentation. It includes:
- Training and evaluation across encoders (`Main_Code.py`)
- Saving quantitative metrics to Excel
- Visualizing model predictions on sample data (`try.py`, `visualize_saved_models.py`)
- Generating comparison bar charts for saved metrics (`BarGraphs.py`)

### Repository structure
- `Main_Code.py`: End-to-end training, evaluation, prediction visualization, and metrics export for U-Net with selectable encoders.
- `BarGraphs.py`: Plots bar charts for Accuracy, Dice, Precision, Recall, Mean IoU to `saved_graph_evaluation/`.
- `visualize_saved_models.py`: Loads previously saved `.h5` models and visualizes predictions on a small dataset batch.
- `try.py`: Loads previously saved `.h5` models and visualizes prediction for a single test image/mask pair.
- `image_dataset/`: Example dataset folder with `images/` and `masks/` subfolders.
- `model_comparison_results.xlsx`: Metrics exported by `Main_Code.py`.
- `model_predictions/`, `saved_model_predictions/`, `saved_graph_evaluation/`, `segmentation_predictions/`: Output folders with images.

### Requirements
- Python 3.9+ recommended
- TensorFlow 2.x (GPU recommended but optional)
- Keras (bundled with TensorFlow 2)
- NumPy, Matplotlib, Pandas
- openpyxl (for writing `.xlsx`)

Install:
```bash
pip install tensorflow numpy matplotlib pandas openpyxl
```

If you have a compatible NVIDIA GPU, install the GPU-enabled TensorFlow according to the official guide (`https://www.tensorflow.org/install`).

### Dataset format
This project expects images and masks arranged in two directory trees that can be consumed by `tf.keras.preprocessing.image_dataset_from_directory`. Typical structure:
```
image_dataset/
  images/
    classA_or_dummy/
      1.png
      2.png
      ...
  masks/
    classA_or_dummy/
      1.png
      2.png
      ...
```

Notes:
- Masks should be single-channel (grayscale) binary masks aligned to images.
- In `Main_Code.py` the directories are set via absolute Windows paths by default. Update them to your local paths.

### Configure paths
Edit the following constants in the scripts to point to your data and models:
- In `Main_Code.py`:
  - `IMAGE_DIR`, `MASK_DIR` (currently set to `D:\DL\image_dataset\images` and `...\masks`)
- In `visualize_saved_models.py`:
  - `IMAGE_DIR`, `MASK_DIR` (dataset roots used for small-batch visualization)
  - `model_paths` dict (absolute paths to saved `.h5` models)
- In `try.py`:
  - `IMAGE_PATH`, `MASK_PATH` (single image/mask)
  - `model_paths` dict (absolute paths to saved `.h5` models)

### Train and evaluate
Runs U-Net with each encoder, evaluates on a test split, saves metrics and prediction montages.
```bash
python Main_Code.py
```
Outputs:
- Trained models: `unet_resnet_model.h5`, `unet_mobilenet_model.h5`, `unet_vgg_model.h5`, `unet_efficientnet_model.h5`
- Metrics Excel: `model_comparison_results.xlsx`
- Prediction grids: `segmentation_predictions/unet_<encoder>_predictions.png`

Metrics recorded per model:
- Accuracy, Dice Coefficient, Precision, Recall, Mean IoU

### Visualize predictions from saved models (single image)
Set `IMAGE_PATH` and `MASK_PATH` in `try.py`, then run:
```bash
python try.py
```
Outputs per encoder in `model_predictions/`:
- `unet_<encoder>_prediction.png`

### Visualize predictions from saved models (small batch)
Configure `IMAGE_DIR`, `MASK_DIR`, and `model_paths` in `visualize_saved_models.py`, then run:
```bash
python visualize_saved_models.py
```
Outputs per encoder in `saved_model_predictions/`:
- `unet_<encoder>_predictions.png`

### Plot comparison bar charts
`BarGraphs.py` uses hard-coded arrays (replace with your actual results or parse the Excel) and saves charts to `saved_graph_evaluation/`.
```bash
python BarGraphs.py
```

### Model architecture
- Encoder: One of ResNet50, MobileNetV2, VGG16, EfficientNetB0 (`include_top=False`, ImageNet weights)
- Decoder: Transposed-convolution U-Net style with skip connections from encoder feature maps
- Output: 1-channel sigmoid mask; trained with `binary_crossentropy`
- Additional metric: Dice Coefficient

### Tips
- Ensure images and masks are aligned by filename and size; the loader normalizes to `[0, 1]` and resizes to `128×128`.
- If your masks are not strictly binary, binarize them before training.
- For larger images, increase `IMG_HEIGHT`, `IMG_WIDTH`, and batch size accordingly (watch VRAM).

### License
This project is provided as-is for educational and research purposes. Add your preferred license here.


