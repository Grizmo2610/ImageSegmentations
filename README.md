# 🖼 IMAGE SEGMENTATION USING U-NET MODEL

## ✨ Description
This project implements an image segmentation model based on the U-Net architecture. U-Net is a convolutional neural network (CNN) specifically designed for biomedical image segmentation. It has gained popularity due to its ability to perform pixel-level predictions in images, enabling precise object segmentation, even with limited training data.

The model works by contracting the input image through an encoder to extract high-level features, then expanding it back through a decoder to predict a mask that segments the image into different classes or regions. The U-Net architecture combines high-level and low-level features via skip connections, which enhances its ability to localize and segment objects accurately.

This project provides a complete pipeline for training, evaluating, and using a U-Net model for various image segmentation tasks, such as medical image analysis, satellite imagery, and more.

## ⚙ Features
- 🧠 U-Net-based image segmentation for precise pixel-level segmentation.
- 📸 Support for custom datasets with easy configuration.
- 💻 Training and inference scripts that enable model training and image segmentation.
- 💾 Model checkpoint saving and loading, making it easy to resume training or deploy pre-trained models.
- 🔧 Data augmentation support to improve model generalization.
- 📈 Logging and visualization of training progress for better monitoring.
- 🖼 Post-processing techniques for refining segmentation output.

## 🖥 Installation
To get started with the U-Net model for image segmentation, follow the steps below:

1. **Clone the repository:**
   Clone the project from GitHub to your local machine:
   ```bash
   git clone https://github.com/your-repo/image-segmentation-unet.git
   cd image-segmentation-unet
   ```

2. **Install dependencies:**
   Install the required Python dependencies listed in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the setup script:**
   Set up the environment and install additional dependencies, if necessary:
   ```bash
   python setup.py install
   ```

4. **Prepare your dataset:**
   For training, use the [COCO 2017 dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset).

5. **Configuration:**
   Modify the necessary parameters directly in the `train.ipynb` notebook, such as dataset paths and model-specific hyperparameters like batch size, learning rate, etc.

---

## 🧑‍💻 Training on Kaggle

In addition to training locally, you can also train the U-Net model directly on Kaggle using a pre-existing notebook. Follow these steps:

1. **Access the Kaggle Notebook:**
   You can access my shared notebook on Kaggle here:
   [Kaggle U-Net Image Segmentation Notebook](https://www.kaggle.com/code/grizmo/imagesegmentations)

2. **Download the Dataset:**
   Before starting training, download the **COCO 2017 dataset** from Kaggle. You can do this via the Kaggle API or download it directly from the dataset page.

3. **Configure the Environment:**
   The required libraries are already installed in the Kaggle notebook. Modify parameters such as dataset paths and model-specific hyperparameters (batch size, learning rate, number of epochs, etc.).

4. **Train the Model:**
   Once configured, run the cells in the notebook to start training the U-Net model. You can monitor the training progress through loss and accuracy plots.

5. **Save the Model:**
   After training is complete, save the trained model and either download it or use it directly for inference.

---

This method allows you to easily train the model without setting up a local environment.

---

## 👩‍💻 Usage

### 🚀 Training the Model
To train the U-Net model, run the following command:
```bash
jupyter notebook train.ipynb
```
Within the notebook, you can modify:
- Dataset paths
- Hyperparameters (learning rate, batch size, number of epochs, etc.)
- Model checkpoints for saving during training

You can monitor the training process through visualizations of training and validation loss, as well as images showing predictions on the validation set.

### 🔍 Running Inference
Once the model is trained, you can use it to perform image segmentation on new images. To run inference, use the following command:
```bash
python script.py --input data/image.png --model models/unet_v1.pth --output output.png
```
In this command:
- `--input` specifies the image you want to segment.
- `--model` specifies the path to the pre-trained U-Net model (e.g., `unet_v1.pth`).
- `--output` specifies the file name where the segmented image will be saved.

The output will be a segmented image with regions clearly identified based on the model’s predictions.

### 🧑‍💻 Hyperparameter Tuning
You can experiment with different hyperparameters such as:
- **Learning Rate:** Affects how quickly the model converges during training.
- **Batch Size:** Determines how many samples are processed before the model’s weights are updated.
- **Epochs:** The number of times the entire dataset is passed through the model.
- **Data Augmentation:** Techniques like flipping, rotation, scaling, and more can be applied to improve model robustness.

These adjustments can be made directly within the `train.ipynb` notebook.

---

## 🛠 How It Works

1. **📃 Data Preparation:**
   - The dataset consists of images and corresponding masks. The images are typically high-resolution and may contain multiple objects or regions of interest.
   - Masks are binary images where each pixel indicates whether it belongs to a target object or the background. These masks are used as ground truth labels during training.

2. **🏆 Training:**
   - U-Net is trained using a typical supervised learning approach, where the model learns to predict the segmentation mask for each input image. It uses a combination of convolutional layers, max pooling, and upsampling layers to extract and refine features.
   - The model is trained using a loss function such as **Dice coefficient** or **Cross-entropy loss**, which helps the model minimize the difference between its predicted masks and the ground truth masks.

3. **🔎 Prediction:**
   - After training, the model is used to predict segmentation masks for new images.
   - The predicted mask is compared to the ground truth mask (if available) to assess the model's performance.

4. **🎨 Post-Processing:**
   - Once the model makes predictions, post-processing steps such as thresholding, morphological operations (e.g., erosion or dilation), and contour detection are used to clean up the segmented image and refine object boundaries.
   - These steps help eliminate noise and small artifacts from the segmented output.

---

## 📒 Understanding U-Net
U-Net is a deep learning architecture specifically designed for image segmentation tasks. It has the following components:
- **📝 Encoder (Contracting Path):** This part of the network extracts high-level features from the input image. It uses convolutional layers and max pooling operations to reduce the spatial dimensions of the image while increasing the depth (number of feature maps).
  
- **🎥 Bottleneck:** This layer connects the encoder and decoder. It represents the most compressed form of the image, containing high-level abstract features.

- **💡 Decoder (Expansive Path):** This part of the network upsamples the features from the bottleneck and combines them with corresponding features from the encoder via skip connections. This allows the model to recover spatial information and produce a detailed segmentation mask.
  
- **Skip Connections:** These connections between the encoder and decoder allow the model to retain fine-grained spatial information, which is crucial for accurate pixel-level segmentation.

---

## 👤 Authors
- **Hoàng Tuấn Tú (Grizmo)** – [GitHub](https://github.com/Grizmo2610)
- **Email:** - hoangtuantu893@gmail.com

## 📚 License
This project is licensed under the MIT License - see the LICENSE file for details.