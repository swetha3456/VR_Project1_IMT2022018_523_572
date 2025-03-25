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
## How to Run the Code

### Steps to Execute
1. Clone the repository and navigate to the project directory.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and save the data in the same directory.

4. Run `part_A.ipynb` to `part_d.ipynb` to view the results.
