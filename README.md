# Laplacian Gradient Consistency Prior for Flash Guided Non-Flash Image Denoising (LGCNet)
- This is the official repository of the paper "Laplacian Gradient Consistency Prior for Flash Guided Non-Flash Image Denoising" from **IEEE Transactions on Image Processing (TIP)**. [[Paper Link]](https://ieeexplore.ieee.org/document/10746360, "Paper Link")


## 1. Environment
- Python >= 3.5
- PyTorch == 1.7.1 is recommended
- opencv-python = =3.4.9.31
- tqdm
- scikit-image == 0.15.0
- scipy == 1.3.1 
- Matlab

## 2. Training and testing dataset
 we adopt the Flash and Ambient Illuminations Dataset (FAID), the Multi-Illumination Dataset (MID) and the DeepFlash Portrait Dataset (DPD) for training and testing
- ***FAID***, we randomly selected 404 flash and non-flash image pairs for training, and used the remaining 12 image pairs for testing. The test image pairs cover image content from different categories.
- ***MID***, captured 1,016  scenes under 25 lighting conditions, of which 984 scenes were used as training set and 30 scenes were used as test set.
- ***DPD***, contains 429 flash and non-flash image pairs. Each pair consists of one photograph
of a subject’s face taken with the camera flash enabled, and the other using a photographic studio-lighting setup. We randomly selected 20 image pairs for testing while using the remaining 409 pairs for training.

All the training and testing images used in this paper can be downloaded from the [[Google Drive Link]](https://drive.google.com/drive/folders/15z2tTMSgYhIQ_QuWjIFUsuquouEy5ZEl?usp=sharing)


## 3. Test
### 🛠️  Clone this repository:
```
    git clone https://github.com/JingyiXu404/LGCNet.git
```
### 🛠️  Download pretrained models:
```
    https://drive.google.com/drive/folders/1a7locw10ahjezyGkPwTgqFgYXfmzEGqt?usp=sharing
```
### 💓  For flash guided non-flash image denoising task
**1. Prepare dataset**: If you do not use same datasets as us, place the test images in `dataset/your_dataname/`.

```
    your_dataname
    └── test_flash
        └── flash
            ├──  1.png 
            ├──  2.png
            └──  3.png
        └── other test datasets
    └── test_nonflash
        └── nonflash
            ├──  1.png 
            ├──  2.png
            └──  3.png
        └── other test datasets
   ```

**2. Run**: `run_test.py`

```
   python run_test.py --iter 3 --sigma 25 --phase 'test' --dataset 'your_dataname'
```
## 4. Contact
If you have any question about our work or code, please email `jingyixu@buaa.edu.cn` .
