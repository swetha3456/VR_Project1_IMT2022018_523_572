# Face Mask Detection, Classification, and Segmentation

## Team members:
1. Swetha Murali - IMT2022018
2. Dyuthi Vivek - IMT2022523
3. PVS Sukeerthi - IMT2022572

## Objective
The objective of this project is to develop a computer vision solution that classifies and segments face masks in images. The project involves using handcrafted features with machine learning classifiers and deep learning techniques to perform classification and segmentation. Additionally, traditional region-based segmentation techniques are used to extract face mask regions.

---
## Dataset

The following datasets are used in this project:

1. **Face Mask Detection Dataset**  
   - A labeled dataset containing images of people with and without face masks.
   - Accessible at: [Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
   - The with mask images can be found at `Face-Mask-Detection/dataset/with_mask/` and the without mask images are at `Face-Mask-Detection/dataset/without_mask/`.
   
2. **Masked Face Segmentation Dataset (MFSD)**  
   - A dataset containing ground truth face masks for segmentation tasks.
   - Accessible at: [MFSD Dataset](https://github.com/sadjadrz/MFSD)
   - The train images can be found at `MSFD/1/face_crop/` and the ground truth labels are at `MSFD/1/face_crop_segmentation/`.

---

## Mask Segmentation Using Traditional Methods

### Preprocessing

- Each image is resized to 128x128 and converted into grayscale before feature extraction.  
- The extracted HOG and LBP histogram features are combined into a single feature vector. 
- The feature vectors are normalized using StandardScaler.
- The dataset is split into training and testing sets with stratified sampling to maintain class balance.

### Methodology and Observations

We extracted features from the resized grayscale face mask images using Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP). HOG captures gradient-based texture information, while LBP encodes local texture patterns. We used these features to train a model that predicts whether or not a mask is worn.

### Hyperparameters and Experiments

We experimented with two models, SVM and XGBoost. SVM with a linear kernel gave an accuracy of 90%, whereas a polynomial kernel gave an accuracy of 91%. 

XGBoost gave us the best accuracy, 92.67%. 

## Results

Classification accuracy: 92.67%

---

## CNN-Based Image Classification with Data Augmentation and Hyperparameter Tuning

**Training and evaluating CNN models** for image classification. The dataset is split into **train, validation, and test sets**. To improve model robustness, **data augmentation** is applied to generate transformed images, making the model invariant to rotations, translations, and other transformations.  
Several CNN models were trained using different **network depths, learning rates, optimizers, and batch sizes** to find the best configuration.

---

### **1. Dataset Preparation**
The dataset was preprocessed and split into:
  - **Training Set**: Used for training the model.
  - **Validation Set**: Used to fine-tune hyperparameters.
  - **Test Set**: Used for final model evaluation.

### **2. Data Augmentation**
To enhance model generalization, the following transformations were applied:
**Rotation** (up to 30 degrees),  **Width and Height Shift** (up to 20%), **Shear Transformation**, **Zoom** (up to 20%), **Horizontal Flip**, **Fill Mode** to handle new pixels after transformation.

### **3. Model Training**
Several CNN architectures were trained, each with:
- **Different number of convolutional layers** (3, 4, 5 layers).
- **Hyperparameter tuning** (batch size, learning rate, optimizer).
- **MaxPooling Layers** were used to reduce spatial dimensions.
- **Dropout** was applied to prevent overfitting.
- The final output layer used **softmax activation** for classification.

### **4. Evaluation Metrics**
The models were evaluated based on  **Accuracy** and **Loss**

---

## **Hyperparameters and Experiments**

The parameters considered are number of layers, batch size, learning rate, optimizer types

- **Different number of convolutional layers** (3, 4, 5 layers).
   ![image](https://github.com/user-attachments/assets/168d86ab-1462-498c-99f6-bd42204c374e)
- **Batch size** (16 ,32, 64).
  ![image](https://github.com/user-attachments/assets/b6758108-966e-4c85-8aa2-92bf17dffc66)
- **Learning rate** (0.001, 0.005, 0.0001)
  ![image](https://github.com/user-attachments/assets/bf53c305-c952-4f7f-997a-767850c02db3)
- **Optimizers** (Adam, SGD, RMSProp).
  ![image](https://github.com/user-attachments/assets/f652d025-d3b6-4fea-9047-387ef49fea4c)


For each parameters 3 models were trained and compared using plots for accuracy and loss.

---

## **Results and Observations**
The test accuracies obtained for with all the parameters is given as"
- **Layers** 3 layers - 0.946, 4 layers - 0.953 , 5 layers - 0.934. 
   - The 4-layer model performed the best (0.953 accuracy), indicating that moderate depth helps extract better features, while excessive depth (5 layers) may cause overfitting or vanishing gradients.
- **Batch sizes** 16 - 0.946, 32 - 0.939 , 64 - 0.961.
   - A larger batch size (64) achieved the highest accuracy (0.961), suggesting better gradient estimation and stable training.
- **Learning rates** 0.001 -0.926 , 0.005 - 0.924, 0.0001 -  0.919 .
   - The best accuracy was at 0.001 because it balanced convergence speed and stability, while lower rates (0.0001) slowed learning and caused underfitting.
- **Optimizers** Adam -0.9488 ,SGD - 0.6366, RMSprop -   0.9512
   - SGD had the lowest accuracy (0.6366) due to slower convergence, sensitivity to learning rate, and lack of adaptive learning, making it less efficient than Adam and RMSprop. Adam or RMSprop optimizers work best.


The best model can be build with 4 layers, 0.001 learning rate, Adam optimizer and batch size of 64. The accuracy obtained was - 0.961.

## Analysis

We observed that traditional feature-based methods, while computationally efficient, rely heavily on handcrafted feature selection, which may not capture the full complexity of mask variations. XGBoost performed slightly better than SVM, suggesting that ensemble methods can leverage weakly correlated features more effectively. However, the CNN-based approach outperformed both, demonstrating that automated feature extraction is more robust against variations in lighting, texture, and positioning. Through hyperparameter tuning, we found that a moderate network depth (4 layers) provided the best balance between feature extraction and overfitting, while larger batch sizes (64) stabilized training. Additionally, Adam and RMSprop optimizers significantly improved convergence compared to SGD, reinforcing the importance of adaptive learning rates in deep learning models.

---

## Mask Segmentation Using Traditional Methods

## Methodology and Observations

### Thresholding-Based Segmentation

We explored Otsu’s thresholding for mask segmentation:

1. **Preprocessing**:
   - Resize the image to `(256, 256)`.
   - Convert the image to grayscale.
   - Apply Gaussian blur to reduce noise.
2. **Otsu’s Thresholding**:
   - Apply Otsu’s method to determine an optimal threshold dynamically.
   - Generate a binary mask based on the threshold value.

While Otsu’s thresholding is effective for simple segmentation tasks, it struggled with varying lighting conditions and complex backgrounds, leading to poor segmentation results.

### Clustering-Based Segmentation

We used K-Means clustering to segment face masks based on color information:

1. **Color Space Conversion**: Convert the image from BGR to LAB color space for better clustering performance.
2. **K-Means Clustering**:
   - Reshape pixel values and apply K-Means clustering with `n_clusters=3`.
   - Assign each pixel to one of the three clusters.
3. **Largest Cluster Selection**:
   - Identify the largest cluster by pixel count.
   - Create a binary mask where pixels belonging to the largest cluster are set to `255`, and others to `0`.

K-Means segmented mask regions without requiring manual/Otsu's thresholding. However, it struggled in cases where the mask color was similar to the skin tone or background.
 

### Watershed Segmentation

A region-based segmentation approach using the Watershed algorithm was employed:

1. **Preprocessing**:
   - Convert the image to grayscale.
   - Apply Canny edge detection.
   - Fill detected edges using binary region filling.
2. **Elevation Map Computation**:
   - Apply Gaussian filtering.
   - Compute an elevation map using the Sobel operator.
3. **Marker Creation**:
   - Assign markers to background, uncertain regions, and foreground (mask area) based on intensity thresholds.
4. **Watershed Algorithm**:
   - Apply Watershed segmentation to separate regions based on elevation gradients.
   - Extract the whitest region as the mask.

Watershed worked better than previous methods. But it required well-defined markers and failed in cases of poor contrast or overlapping regions, leading to limited segmentation performance.


### GrabCut Segmentation

GrabCut, an iterative graph-cut-based segmentation method, was used:

1. **Preprocessing**:
   - Resize the image.
   - Define a bounding box slightly inside the image border.
2. **GrabCut Application**:
   - Initialize a binary mask and background/foreground models.
   - Apply the GrabCut algorithm iteratively with `GC_INIT_WITH_RECT`.
3. **Binary Mask Extraction**:
   - Convert the result to a binary mask, marking foreground pixels as `255` and background as `0`.

GrabCut produced better results than the previous methods by leveraging iterative refinement. However, it still exhibited challenges in distinguishing face masks as compared to deep learning-based methods.



### Results

#### Thresholding

- **Mean IoU:** 0.2800
- **Mean Dice:** 0.0015 

#### Clustering
- **Mean IoU:** 0.3406  
- **Mean Dice:** 0.0018 

#### Watershed
- **Mean IoU:** 0.3168  
- **Mean Dice:** 0.4373

#### GrabCut
- **Mean IoU:** 0.5569 
- **Mean Dice:** 0.6968  

---

## Mask Segmentation Using U-Net

### Methodology

1. **Data Preprocessing**: The images were resized and pixel values were normalized.

2. **Feature Extraction**: We use a U-Net architecture to extract features at different spatial resolutions using an encoder-decoder structure with skip connections.

3. **Model Training**: The U-Net model is trained using a loss function such as binary cross-entropy loss function using an Adam optimizer.

4. **Segmentation Techniques**: The trained model predicts binary masks, which are post-processed to refine segmentations using thresholding.

### Hyperparameters and Experiments

- **Architecture**: The U-Net architecture used involved five successive double convolution layers followed by max-pooling layers with stride 2 for the encoder, and four transpose convolutions with skip connections. We tried increasing the complexity of the architecture by adding more layers, but there was no significant change in accuracy.

- **Epochs**: The network was trained for 8 epochs as there seemed to be negligible change in validation loss beyond this point.

- **Learning Rate**: We found a learning rate of 1e-3 to be optimal. For the optimal batch size of 32, we obtained the following scores for Dice and IoU on the test dataset:
   
   1. Learning Rate: 1e-2, Dice: 0.9417, IoU: 0.7997
   2. Learning Rate: 1e-3, Dice: 0.9558, IoU: 0.8259
   3. Learning Rate: 1e-4, Dice: 0.9529, IoU: 0.8129

- **Batch Size**: The optimal batch size was 32. For the optimal learning rate of 1e-3, we obtained the following scores for Dice and IoU on the test dataset:
   
   1. Batch Size: 32, Dice: 0.9558, IoU: 0.8259
   2. Batch Size: 64, Dice: 0.9529, IoU: 0.8183

### Results

- **Best IoU:** 0.9558
- **Best Dice Score:** 0.8259

## Analysis

U-Net outperformed traditional methods, achieving 48% higher IoU than GrabCut due to its encoder-decoder structure and skip connections. Hyperparameter tuning played a crucial role, with a learning rate of 1e-3 and a batch size of 32 proving optimal. Training beyond 8 epochs showed diminishing returns, preventing overfitting. 

## How to Run the Code

### Steps to Execute
1. Clone the repository and navigate to the project directory.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and save the data in the same directory.

4. Run all the `.ipynb` notebooks to view the results.

