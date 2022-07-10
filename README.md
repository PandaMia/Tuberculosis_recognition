# Tuberculosis recognition  

The PyTorch neural network model for tuberculosis classification.

## Data

[Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) from Kaggle.

## Training

To train model you need:
* Download the Tuberculosis dataset from the link above
* Clone repository and install requirements:
  * ```git clone https://github.com/PandaMia/Tuberculosis_recognition.git```
  * ```cd Tuberculosis_recognition```
  * ```pip install -r requirements.txt```
* Run data preparation:  
  * ```python data_preparation.py C:\Path\To\Archive\TB_database.zip```  
* Train model:
  * ```python train.py --epochs 20 --lr 0.0003 --bs 32```
  
The weights of the trained model will be saved in the file ```./source/model/model_weights.pth```
  
To evaluate model on test data you need:
* Run evaluation script:
  * ```python evaluation.py --bs 32```

## Inference

To launch model on inference you could train your own model or download weights:  
[https://drive.google.com/file/d/1wKYaVTo1j_Nd_6E-hMMwTVWCSUwYEzyZ/view?usp=sharing](https://drive.google.com/file/d/1wKYaVTo1j_Nd_6E-hMMwTVWCSUwYEzyZ/view?usp=sharing)  

The downloaded weights must be placed on the path ```./source/model```

Then run streamlit app:
* ```streamlit run app.py```

In the app you could upload your image or choose preset image to test the model.

![](https://github.com/PandaMia/Tuberculosis_recognition/blob/dev/source/data/images/app_example.jpg)

## Research

### Data preparation 

I noticed that normal images have a spectrum of grayscale. But tuberculosis images can have different color spectrums.  
In addition half of the tuberculosis images have only one channel.

![](https://github.com/PandaMia/Tuberculosis_recognition/blob/dev/source/data/images/different_colors.png)

So that the neural network doesn't use color as a feature for classification I bring images to a single color spectrum. I do this by transforming all images to one channel.

This dataset has a class imbalance: 3500 normal images and only 700 TB images. I take this into account when:
* Training models: data augmentation; increase loss for TB class
* Evaluate models. On this dataset accuracy metric is not suitable. The best choice is F1-score which is more reliable. Baseline accuracy for this task is 83%.

### Model

I am using a CNN model with four convolutional blocks (two convolutional layers in each block) and four fc-layers for classification.  

The loss function is BCEWithLogitsLoss. This binary crossentropy allow to increase the weights for imbalanced class.  

I also use a callback to reduce learning rate when a plateau is reached.

### Results

Accuracy on the test dataset is ```99.29%```, f1-score is ```97.87%```.   

Confusion matrix:  
![](https://github.com/PandaMia/Tuberculosis_recognition/blob/dev/source/data/images/confusion_matrix.png)

These are the same results as reached on validation dataset during training.

I have a suggestion that achieved accuracy probably higher due to specific of tuberculosis scans.
* Different color spectrums of TB and normal images (I tried to solve this problem when preparing the data)
* Some TB images have white rectangles while normal images do not
* A significant part of the TB images are labeled "L" and "R"

![](https://github.com/PandaMia/Tuberculosis_recognition/blob/dev/source/data/images/labeled.jpg)

These specifics aren't significant for tuberculosis classification. But it can be the feature for classification because only TB images have this specificity. The model trained on this dataset may perform worse on images without described features.

The cause of differences between classes is probably the different resources (medical organizations) for data collection.
