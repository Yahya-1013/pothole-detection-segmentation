# Pothole Detection using Semantic Segmentation

A computer vision project that detects and segments potholes in road images using deep learning-based semantic segmentation.

## Overview

This project implements a semantic segmentation model to identify and localize potholes in road images. Using state-of-the-art deep learning architectures like U-Net and DeepLabV3, the model can accurately segment pothole regions for road maintenance planning.

## Features

- 🛣️ **Pothole Detection**: Accurate segmentation of pothole regions in road images
- 🤖 **Multiple Architectures**: U-Net and DeepLabV3 implementations
- 📊 **Data Pipeline**: Automated data loading and preprocessing
- 🎯 **Real-time Inference**: Deploy model for live road monitoring
- 📈 **Comprehensive Evaluation**: Metrics like IoU, Dice, and Pixel Accuracy
- 📝 **Training Logs**: TensorBoard integration for visualization

## Project Structure

```
pothole-detection-segmentation/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── download_dataset.py
├── models/
│   ├── __init__.py
│   ├── unet.py
│   ├── deeplabv3.py
│   └── segmentation_model.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_evaluation.ipynb
├── tests/
│   ├── __init__.py
│   └── test_models.py
├── outputs/
│   ├── models/
│   ├── logs/
│   └── predictions/
└── .gitignore
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Yahya-1013/pothole-detection-segmentation.git
cd pothole-detection-segmentation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Download Dataset
```bash
cd data
python download_dataset.py
cd ..
```

### Training

```bash
python src/train.py --config config/config.yaml
```

### Evaluation

```bash
python src/evaluate.py --model outputs/models/best_model.pth --dataset data/processed/test/
```

### Inference on New Images

```bash
python src/inference.py --model outputs/models/best_model.pth --image path/to/image.jpg
```

## Model Architectures

### U-Net
- Encoder-decoder architecture with skip connections
- Excellent for medical and road damage segmentation
- Parameters: ~7.8M

### DeepLabV3
- Atrous Spatial Pyramid Pooling (ASPP)
- Better for capturing multi-scale features
- Parameters: ~39M

## Dataset

The project uses pothole/road damage datasets:
- **RDD2020**: Road Damage Detection Dataset
- **Custom datasets**: Support for custom annotated datasets in COCO or Pascal VOC format

## Performance Metrics

- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth
- **Dice Coefficient**: Harmonic mean of precision and recall
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Mean Absolute Error (MAE)**: Average prediction error

## Results

Typical performance on validation set:
- IoU: ~85%
- Dice Score: ~92%
- Pixel Accuracy: ~96%

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- U-Net: Ronneberger et al., 2015
- DeepLabV3: Chen et al., 2017
- RDD2020 Dataset: https://github.com/sekilab/RDD2020

## Contact

For questions or suggestions, please create an issue in the repository.

---

**Happy coding!** 🚀
