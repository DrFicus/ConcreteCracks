# Concrete Crack Detection Model (Evaluation)

### Introduction
This project contains a machine learning model for detecting concrete cracks. The model is trained to categorize images into 'Negative' (no crack) or 'Positive' (crack present) classes. It is designed to evaluate its performance on a provided dataset and categorize the failed cases for further analysis.

### Directory Structure
- `concrete_crack_model_complete.pth`: The trained model file.
- `eval.py`: Python script for evaluating the model on a dataset.
- `Evaluation Dataset`: Folder containing the dataset for evaluation, divided into 'Negative' and 'Positive' subfolders.
- `Failed`: Folder to store images that were incorrectly classified by the model, divided into 'Failed Negative' and 'Failed Positive'.
- `README.md`: Documentation file (this file).
- `requirements.txt`: Required libraries and dependencies for the project.

### eval.py
The `eval.py` script is used to evaluate the model's performance. Key functionalities include:
- **Device Selection**: Automatically selects the best available processing device (MPS, CUDA, CPU).
- **Image Processing**: Transforms images into the format required by the model.
- **Model Loading**: Loads the pre-trained model.
- **Prediction**: Predicts the category of each image in the evaluation dataset.
- **Misclassification Handling**: Copies misclassified images into 'Failed Negative' or 'Failed Positive' for further analysis.
- **Performance Metrics**: Calculates and displays the accuracy of the model.

### How to Run
1. Ensure all dependencies in `requirements.txt` are installed.
2. Place the evaluation dataset in the 'Evaluation Dataset' directory.
3. Run `python eval.py` to start the evaluation process.

### Notes
- The script will clear the 'Failed' directory before each run to ensure only the current evaluation's failed cases are stored.
