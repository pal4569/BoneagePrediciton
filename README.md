# Bone Age Prediction with CNN Model

This ML project is intended to correctly identify the age of a child from an X-ray of their hand.

Divided into three main sections:

    serializeData.py: Prepares the dataset for training and evaluation.
    model.py: Trains the SCN model using the prepared dataset.
    evaluate.py: Evaluates the trained model's performance.

### Dataset

The model is trained using over 12,000 radiology images provided by Kaggle's RSNA Bone Age dataset. You can find more information about the dataset [here](https://www.kaggle.com/datasets/kmader/rsna-bone-age/data).

### Current Status

This project is a work in progress. While it is entirely functional, it yields an MAE ~60 months on average. According to Kaggle, the best models have MAEs of 5-7 months.

### Requirements

To run this project on your local machine, ensure you have the dependencies listed in dependencies.txt installed.
After downloading the dataset from Kaggle, I moved both the training and validation datasets into a "data" directory.
