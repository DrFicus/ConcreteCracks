# Concrete Crack Detection Model (Training)

## Introduction
This project utilizes deep learning techniques to classify images of concrete into two categories: cracked (positive) and not cracked (negative). A pre-trained ResNet50 model, adapted for binary classification, is employed for this task.

## Requirements
To run the project, please install the necessary Python packages listed in `requirements.txt`. 

## Dataset
The dataset is acquired from [Kaggle: Concrete Crack Images for Classification](https://www.kaggle.com/arnavr10880/concrete-crack-images-for-classification). It comprises 40,000 images, split evenly into two folders: 'Negative' and 'Positive'. Each folder contains 20,000 images representing concrete surfaces without and with cracks, respectively.

## Project Structure
- `concrete_crack_model_complete.pth`: A saved PyTorch model in its entirety (output from the training process).
- `concrete_crack_model_state_dict.pth`: The state dictionary of the trained PyTorch model (output from the training process).
- `ConcreteCracks.ipynb`: Jupyter notebook containing the entire workflow, including data loading, preprocessing, model training, evaluation, and saving.
- `requirements.txt`: Required Python packages for running the notebook.
- `README.md`: Documentation file (this file).
- `Dataset`: Directory containing the dataset with two subfolders (`Negative` and `Positive`).

## Usage
1. Install the required packages: `pip install -r requirements.txt`.
2. Run the `ConcreteCracks.ipynb` notebook. This notebook contains the following steps:
   - **Data Loading**: Loading and transforming the dataset from the 'Dataset' folder.
   - **Data Preprocessing**: Applying transformations like resizing and normalization to the images.
   - **Model Training**: Training the ResNet50 model on the dataset.
   - **Model Evaluation**: Evaluating the model's performance on a validation set.
   - **Model Saving**: Saving the trained model to 'concrete_crack_model_complete.pth'.
3. The trained model can be loaded from `concrete_crack_model_complete.pth` for further use or inference. The notebook also includes instructions for model evaluation and saving the model's state.

## Note
This project is for educational and research purposes. The model's accuracy and effectiveness in real-world scenarios may vary.
