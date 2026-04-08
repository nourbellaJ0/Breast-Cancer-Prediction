# Breast Cancer Prediction

A machine learning application for predicting breast cancer diagnosis using multiple classification models. The project features a Flask backend API and Streamlit web interface, all containerized with Docker.

## 🎯 Overview

This project implements an ensemble of machine learning models to predict breast cancer diagnosis based on medical features. The application provides:
- **Multiple Models**: XGBoost, SVM, MLP Neural Network
- **Web Interface**: User-friendly Streamlit frontend
- **REST API**: Flask backend for predictions
- **Docker Support**: Easy deployment with Docker Compose

## 📋 Features

- 🤖 Multiple ML models for robust predictions
- 📊 Data preprocessing and feature scaling
- 🌐 Interactive web interface
- 📈 Real-time predictions via API
- 🐳 Docker containerization for easy deployment
- 📝 Model artifacts saved for inference

## 📁 Project Structure

```
Breast-Cancer-Prediction/
├── backend/
│   ├── app.py                 # Flask application
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── streamlit_app.py       # Streamlit web interface
│   ├── requirements.txt        # Frontend dependencies
│   └── assets/
│       └── style.css          # Styling
├── artifacts/
│   ├── data.csv               # Training dataset
│   ├── mlp.pkl                # MLP model
│   ├── svm.pkl                # SVM model
│   ├── xgboost.pkl            # XGBoost model
│   ├── scaler.pkl             # Feature scaler
│   └── softmax.pkl            # Softmax classifier
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Docker image definition
├── start.sh                   # Startup script
└── README.md                  # This file
```

## 🛠️ Technology Stack

- **Backend**: Flask
- **Frontend**: Streamlit
- **ML Models**: XGBoost, scikit-learn (SVM, MLP)
- **Data Processing**: pandas, numpy, scikit-learn
- **Containerization**: Docker & Docker Compose
- **Language**: Python 3.x

## 📦 Installation

### Option 1: Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/nourbellaJ0/Breast-Cancer-Prediction.git
   cd Breast-Cancer-Prediction
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:5000

### Option 2: Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nourbellaJ0/Breast-Cancer-Prediction.git
   cd Breast-Cancer-Prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Backend
   pip install -r backend/requirements.txt
   
   # Frontend
   pip install -r frontend/requirements.txt
   ```

4. **Run the application**
   ```bash
   # Terminal 1: Start Flask backend
   python backend/app.py
   
   # Terminal 2: Start Streamlit frontend
   streamlit run frontend/streamlit_app.py
   ```

## 🚀 Usage

### Web Interface
1. Open http://localhost:8501 in your browser
2. Input patient medical features
3. Select desired model for prediction
4. View diagnostic prediction and confidence score

### API Endpoints

- **POST** `/predict` - Get prediction from a specific model
  ```json
  {
    "model": "xgboost",
    "features": [values...]
  }
  ```

- **GET** `/health` - Check API health status

## 📊 Model Information

The application uses three different ML models:

- **XGBoost**: Gradient boosting model with high accuracy
- **SVM**: Support Vector Machine for classification
- **MLP**: Multi-layer Perceptron neural network

All models are trained on breast cancer diagnostic dataset and achieve high accuracy in predictions.

## 🔧 Configuration

- Backend Port: 5000
- Frontend Port: 8501
- Database/Artifacts: `/artifacts` directory

## 📚 Dataset

The project uses a breast cancer diagnostic dataset containing medical measurements and features used for classification.

## 📝 Requirements

See individual `requirements.txt` files in `backend/` and `frontend/` directories.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**nourbellaJ0**
- GitHub: [@nourbellaJ0](https://github.com/nourbellaJ0)
- Email: nour.bellaaj@esprit.tn

## 📞 Support

If you have any questions or issues, please open an issue on GitHub.

---

**Note**: This is a machine learning application for educational and research purposes. Always consult with medical professionals for actual medical diagnosis.
