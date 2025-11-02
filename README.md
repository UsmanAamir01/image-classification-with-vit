# 🍽️ FoodVision AI - Food Image Classifier

A professional web application for food image classification using Vision Transformer (ViT) deep learning model. Built with Streamlit and Hugging Face Transformers.

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/UsmanAamir01/image-classification-with-vit)

## 🔗 Links

- **GitHub Repository**: [https://github.com/UsmanAamir01/image-classification-with-vit](https://github.com/UsmanAamir01/image-classification-with-vit)
- **Live Demo**: Coming soon on Streamlit Cloud

## 🌟 Features

- **Advanced AI Recognition**: Vision Transformer (ViT) architecture for accurate food classification
- **101 Food Categories**: Recognizes a wide variety of cuisines and dishes
- **Professional UI**: Modern, responsive interface with blue and black theme
- **Real-time Predictions**: Instant analysis with confidence scores
- **Top-5 Results**: View detailed predictions with probability percentages
- **Sample Images**: Try pre-loaded food images for quick testing
- **Interactive Visualizations**: Progress bars, confidence badges, and metrics

## 🎯 Supported Food Categories

Pizza, Sushi, Hamburger, Cheesecake, Pasta, Salad, Ice Cream, Tacos, Ramen, Steak, Pancakes, and 90+ more food types from various cuisines worldwide.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/UsmanAamir01/image-classification-with-vit.git
cd image-classification-with-vit
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the model** (if not included)

The trained model is included in the repository via Git LFS. If you need to manually download it:

- Model files are located in `results/vit-food-classifier-final/`
- Total size: ~344 MB (stored with Git LFS)

### Running the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 🎓 Training

To train your own model or reproduce the results:

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook image-classification-with-vit.ipynb
   ```

2. Follow the notebook cells to:
   - Load and preprocess the Food-101 dataset
   - Fine-tune the Vision Transformer model
   - Evaluate performance
   - Save the trained model

## 📦 Project Structure

```
image-classification-with-vit/
│
├── app.py                          # Main Streamlit application
├── image-classification-with-vit.ipynb  # Training notebook
├── requirements.txt                # Python dependencies
├── .gitignore                     # Git ignore file
├── .gitattributes                 # Git LFS configuration
├── LICENSE                        # MIT License
├── README.md                      # This file
│
├── archive/
│   ├── images/                    # Sample food images (101 categories)
│   └── meta/                      # Metadata files
│
└── results/
    ├── label_mappings.json        # Category label mappings
    └── vit-food-classifier-final/  # Trained model files
        ├── config.json
        ├── model.safetensors       # Model weights (344 MB)
        ├── preprocessor_config.json
        └── label_mappings.json
```

## 🎨 UI Features

- **Hero Header**: Eye-catching gradient header with app branding
- **Dual Column Layout**: Upload on left, results on right
- **Gradient Card Design**: Professional cards with white text on blue/black backgrounds
- **Animated Elements**: Smooth fade-in and slide-in animations
- **Progress Indicators**: Real-time feedback during analysis
- **Confidence Badges**: Color-coded badges (High/Medium/Low confidence)
- **Performance Stats**: Model statistics in sidebar

## 🤖 Model Details

- **Architecture**: Vision Transformer (ViT) - `google/vit-base-patch16-224-in21k`
- **Base Model**: Pre-trained on ImageNet-21k
- **Fine-tuned On**: Food-101 Dataset
- **Hidden Size**: 768 dimensions
- **Attention Heads**: 12 heads
- **Layers**: 12 transformer blocks
- **Patch Size**: 16×16 pixels
- **Parameters**: ~86M
- **Input Size**: 224×224 RGB images
- **Framework**: Hugging Face Transformers
- **Classes**: 101 food categories

## 💡 Usage Tips

1. **Use clear, well-lit images** for best results
2. **Center the food in the frame**
3. **Avoid cluttered backgrounds**
4. **Single food item works best**
5. **Try different angles** if confidence is low

## 📊 Performance

- **Fast Inference**: Quick predictions in seconds
- **High Accuracy**: Trained on extensive food dataset
- **Robust**: Handles diverse food types and presentations
- **Modern Architecture**: State-of-the-art transformer model

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - Web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Model library
- [Pillow](https://python-pillow.org/) - Image processing

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👨‍💻 Author

**Usman Aamir**

- GitHub: [@UsmanAamir01](https://github.com/UsmanAamir01)
- Repository: [image-classification-with-vit](https://github.com/UsmanAamir01/image-classification-with-vit)

Built with ❤️ using Streamlit & Hugging Face

## 🙏 Acknowledgments

- Food-101 Dataset
- Hugging Face community
- Streamlit team

## 📧 Contact

For questions or feedback, please:

- Open an issue on [GitHub](https://github.com/UsmanAamir01/image-classification-with-vit/issues)
- Visit the repository: [https://github.com/UsmanAamir01/image-classification-with-vit](https://github.com/UsmanAamir01/image-classification-with-vit)

---

**Note**: This application requires the trained model files to be present in the `results/vit-food-classifier-final/` directory. Make sure to download or train the model before running the app.
