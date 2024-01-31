import os
import torch
import random
import shutil
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score

class ImagePredictor:
    def __init__(self, model_path):
        self.device = self._select_device()
        self.transform = self._define_transform()
        self.model = self._load_model(model_path)

    def _select_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _define_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet standard values
        ])

    def _load_model(self, model_path):
        model = torch.load(model_path)
        model = model.to(self.device)
        model.eval()
        return model

    def predict_image(self, image_path):
        image = Image.open(image_path)

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = self.transform(image)
        image = image.unsqueeze(0) # Batch size of 1
        image = image.to(self.device)
        with torch.no_grad(): # Inference not training
            output = self.model(image)
            prediction = torch.argmax(output, dim=1).item()
        return prediction

class ImageEvaluator:
    def __init__(self, image_predictor, test_images_path, failed_output_images_path):
        self.image_predictor = image_predictor
        self.test_images_path = test_images_path
        self.failed_output_images_path = failed_output_images_path
        self.categories = ['Negative', 'Positive']
        self.labels = []
        self.predictions = []

    def evaluate(self, sample_size=50):
        for index, category in enumerate(self.categories):
            category_path = os.path.join(self.test_images_path, category)
            all_images = [img for img in os.listdir(category_path) if img.lower().endswith(('.jpg', '.png'))]
            sampled_images = random.sample(all_images, min(len(all_images), sample_size))

            for image_file in sampled_images:
                image_path = os.path.join(category_path, image_file)
                prediction = self.image_predictor.predict_image(image_path)
                self.predictions.append(prediction)
                self.labels.append(index)
                print(f"Image: {image_file}, True label: {index}, Predicted: {prediction}") 

                if prediction != index:
                    target_path = self._get_failed_path(index)
                    shutil.copy(image_path, target_path)

        accuracy = accuracy_score(self.labels, self.predictions)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def _get_failed_path(self, index):
        target_directory = 'Failed Positive' if index == 1 else 'Failed Negative'
        return os.path.join(self.failed_output_images_path, target_directory)

    @staticmethod
    def clear_directory(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:

                print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == "__main__":
    
    image_predictor = ImagePredictor('concrete_crack_model_complete.pth')
    image_evaluator = ImageEvaluator(image_predictor, './Evaluation Dataset', './Failed')

    ImageEvaluator.clear_directory('./Failed/Failed Positive')
    ImageEvaluator.clear_directory('./Failed/Failed Negative')

    sample_size = 50
    image_evaluator.evaluate(sample_size)
