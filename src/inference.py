import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
from models import UNet, DeepLabV3
from src.utils import SegmentationMetrics


class Inference:
    """Inference module for pothole detection"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model_path (str): Path to trained model checkpoint
            device (str): Device to run inference on
        """
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint['config']
        
        # Build and load model
        if self.config['model']['name'] == 'unet':
            self.model = UNet(
                in_channels=self.config['model']['in_channels'],
                out_channels=self.config['model']['out_channels']
            )
        else:
            self.model = DeepLabV3(
                in_channels=self.config['model']['in_channels'],
                out_channels=self.config['model']['out_channels']
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.image_size = self.config['data']['image_size']
        print(f"Model loaded from: {model_path}")
    
    def preprocess(self, image):
        """Preprocess image"""
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # To tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image
    
    def postprocess(self, output, threshold=0.5):
        """Postprocess model output"""
        mask = torch.sigmoid(output).squeeze().cpu().detach().numpy()
        mask = (mask > threshold).astype(np.uint8) * 255
        return mask
    
    def predict(self, image_path, threshold=0.5):
        """Predict pothole mask for an image"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape
        
        # Preprocess
        preprocessed = self.preprocess(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(preprocessed)
        
        # Postprocess
        mask = self.postprocess(output, threshold)
        
        # Resize to original size
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
        
        return mask, image
    
    def visualize_prediction(self, image_path, threshold=0.5, output_path=None):
        """Visualize prediction with overlay"""
        mask, image = self.predict(image_path, threshold)
        
        # Resize image to match mask
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
        
        # Create overlay
        overlay = image.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red for potholes
        
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Convert to BGR for cv2.imwrite
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Prediction saved to: {output_path}")
        
        return result, mask


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output')
    parser.add_argument('--threshold', type=float, default=0.5, help='Segmentation threshold')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    inference = Inference(args.model, device=args.device)
    result, mask = inference.visualize_prediction(args.image, args.threshold, args.output)
    print(f"Pothole pixels: {(mask > 0).sum()}")
