# Bone Age Prediction with SCN Model

This ML projected is intended to correctly identify the age of a child from an X-ray of their hand.

The project is divided into three main sections:

    serializeData.py: Prepares the dataset for training and evaluation.
    model.py: Trains the SCN model using the prepared dataset.
    evaluate.py: Evaluates the trained model's performance.

### Dataset

The model is trained using over 12,000 radiology images provided by Kaggle's RSNA Bone Age dataset. You can find more information about the dataset [here](https://www.kaggle.com/datasets/kmader/rsna-bone-age/data).

### Current Status

This project is a work in progress. While it is entirely functional, it is not very accurate, yielding unideal validation MAE. 

### Requirements

To run this project on your local machine, ensure you have the dependencies listed in dependencies.txt installed.
