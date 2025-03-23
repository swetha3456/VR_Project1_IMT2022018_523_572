# Face Mask Detection, Classification, and Segmentation

## Objective
The objective of this project is to develop a computer vision solution that classifies and segments face masks in images. The project involves using handcrafted features with machine learning classifiers and deep learning techniques to perform classification and segmentation. Additionally, traditional region-based segmentation techniques are used to extract face mask regions.

---
## Dataset

The following datasets are used in this project:

1. **Face Mask Detection Dataset**  
   - A labeled dataset containing images of people with and without face masks.
   - Accessible at: [Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
   
2. **Masked Face Segmentation Dataset (MFSD)**  
   - A dataset containing ground truth face masks for segmentation tasks.
   - Accessible at: [MFSD Dataset](https://github.com/sadjadrz/MFSD)

---
## Mask Segmentation Using Traditional Methods

## Methodology and Observations

### Thresholding-Based Segmentation

We explored thresholding techniques for mask segmentation, including basic and adaptive thresholding:

1. **Basic Thresholding**:
   - Convert the image to grayscale.
   - Apply a fixed threshold (`127/255`) to segment mask regions.
   - Invert the binary mask for better visualization.
2. **Adaptive Thresholding**:
   - Convert the image to grayscale.
   - Apply Adaptive Gaussian Thresholding with block size `11` and constant `2`.
   - This method adjusts thresholds dynamically based on local pixel intensity variations.

The IoU and dice results were extremely poor.


### Clustering-Based Segmentation

We used K-Means clustering to segment face masks based on color information:

1. **Color Space Conversion**: The image is converted from BGR to LAB color space, which is more effective for clustering due to its perceptual uniformity.
2. **K-Means Clustering**: The LAB pixel values are reshaped and clustered into `n_clusters=3` using K-Means.
3. **Largest Cluster Selection**: After clustering, the largest cluster is identified based on pixel count.
4. **Binary Mask Creation**: A binary mask is generated where pixels belonging to the largest cluster are set to 255, and others to 0.

K-Means effectively separates different color regions without requiring manual thresholding. However, it  struggled with lighting variations or images where the face and mask are similarly coloured.


Mean IoU: 0.3406

Mean dice: 0.0018

### Watershed Segmentation

A region-based segmentation approach using the Watershed algorithm was followed. The following steps were done:

1. **Preprocessing**:
   - Convert the image to grayscale.
   - Detect edges using the Canny edge detector.
   - Fill detected edges using binary region filling.
2. **Elevation Map Computation**:
   - Apply Sobel filtering to compute an elevation map that highlights edges and gradients in the image.
3. **Marker Creation**:
   - Define foreground and background markers based on pixel intensity thresholds.
   - Assign three marker values: background, transition, and foreground (mask region).
4. **Watershed Algorithm**:
   - Apply the watershed segmentation to separate different regions based on the elevation map.

Watershed segmentation required well-defined markers and failed in cases of poor contrast or overlapping regions.

Mean IoU: 0.3333
Mean Dice: 0.4452

### Results

IoU for Clustering & Watershed: ~0.3 (both methods performed similarly in overlap with ground truth).

Dice Score: Watershed performed better (~0.4). It captured the shape of the mask regions more effectively than clustering.

---
## How to Run the Code

### Steps to Execute
1. Clone the repository and navigate to the project directory.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and save the data in the same directory.

4. Run `segmentation.ipynb` and `classification.ipynb` to view the results.
