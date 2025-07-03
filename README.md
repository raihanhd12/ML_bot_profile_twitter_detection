# 🤖 Twitter Bot Detection using Machine Learning

A comprehensive machine learning project for detecting bot accounts on Twitter using various behavioral and profile features.

## 📊 Project Overview

This project implements an advanced bot detection system that analyzes Twitter account characteristics to distinguish between genuine human accounts and automated bot accounts. The model achieves high accuracy by examining multiple behavioral patterns and profile features.

## 🎯 Features

### **Data Analysis & Visualization**

- Comprehensive exploratory data analysis (EDA)
- Statistical comparison between bot and human accounts
- Feature correlation analysis
- Distribution plots and visualizations

### **Feature Engineering**

- **Profile Features**: Bio length, username characteristics, profile customization
- **Behavioral Features**: Follower-to-following ratio, tweet frequency, account age
- **Verification Features**: Profile verification status, location info, geo-enabling
- **Activity Metrics**: Favorites count, statuses count, average tweets per day

### **Machine Learning Models**

- **Tree-based Models**: Random Forest, XGBoost, LightGBM, Extra Trees
- **Ensemble Methods**: Gradient Boosting, AdaBoost
- **Linear Models**: Logistic Regression, SVM
- **Probabilistic Models**: Naive Bayes
- Cross-validation with stratified k-fold
- Model performance comparison and selection

## 📁 Project Structure

```
bot-detection-twitter/
├── 0_dataaset_ai.ipynb          # Initial dataset exploration
├── 1_dev_1.ipynb               # Main development notebook
├── 2_dev.ipynb                 # Alternative development approach
├── 2_model_inspect.ipynb       # Model inspection and analysis
├── requirements.txt            # Python dependencies
├── data/                       # Dataset directory
├── twitter_bot_detection_model/ # Saved model artifacts
│   ├── best_model.pkl          # Trained model
│   ├── scaler.pkl              # Feature scaler
│   └── model_metadata.json     # Model metadata
└── README.md                   # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/bot-detection-twitter.git
cd bot-detection-twitter
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 📚 Dependencies

```
datasets
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
lightgbm
wordcloud
jupyter
joblib
```

## 🚀 Usage

### **Quick Start**

1. **Run the main notebook**

```bash
jupyter notebook 1_dev_1.ipynb
```

2. **Load and use the trained model**

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('twitter_bot_detection_model/best_model.pkl')
scaler = joblib.load('twitter_bot_detection_model/scaler.pkl')

# Example prediction
sample_data = {
    'favourites_count': 1000,
    'followers_count': 500,
    'friends_count': 200,
    'statuses_count': 1500,
    'average_tweets_per_day': 2.5,
    'account_age_days': 365,
    'follower_following_ratio': 2.5,
    'bio_length': 120,
    'username_length': 12,
    'username_digit_count': 2,
    'has_custom_profile_image': 1,
    'has_custom_background': 1,
    'has_location': 1,
    'is_default_profile': 1,
    'is_geo_enabled': 1,
    'is_verified': 0
}

# Make prediction
prediction = model.predict([list(sample_data.values())])
probability = model.predict_proba([list(sample_data.values())])

print(f"Prediction: {'Bot' if prediction[0] == 1 else 'Human'}")
print(f"Confidence: {max(probability[0]):.3f}")
```

## 🔍 Dataset

The project uses the **"nahiar/twitter_bot_detection"** dataset from Hugging Face, which contains:

- **Size**: Multiple thousands of Twitter accounts
- **Features**: 20+ account characteristics
- **Labels**: Binary classification (Human/Bot)
- **Source**: Real Twitter account data

### **Key Features Used**

#### **Numeric Features**

- `favourites_count`: Number of tweets liked
- `followers_count`: Number of followers
- `friends_count`: Number of accounts followed
- `statuses_count`: Total tweets posted
- `average_tweets_per_day`: Daily tweet frequency
- `account_age_days`: Account age in days
- `follower_following_ratio`: Followers to following ratio
- `bio_length`: Length of profile bio
- `username_length`: Length of username
- `username_digit_count`: Number of digits in username

#### **Binary Features**

- `has_custom_profile_image`: Custom profile picture
- `has_custom_background`: Custom background image
- `has_location`: Location information provided
- `is_default_profile`: Using default profile settings
- `is_geo_enabled`: Geo-location enabled
- `is_verified`: Account verification status

## 📈 Model Performance

### **Best Model**: XGBoost/Random Forest (varies by run)

| Metric        | Score  |
| ------------- | ------ |
| **Accuracy**  | ~0.96+ |
| **Precision** | ~0.95+ |
| **Recall**    | ~0.94+ |
| **F1-Score**  | ~0.95+ |
| **AUC-ROC**   | ~0.98+ |

### **Key Insights**

- **Follower-to-following ratio** is highly predictive
- **Account age** and **tweet frequency** are strong indicators
- **Profile customization** features help distinguish bots
- **Username characteristics** (length, digits) are significant

## 🔬 Analysis Highlights

### **Bot vs Human Characteristics**

**Bots typically have:**

- Higher follower-to-following ratios
- Shorter or generic bios
- More digits in usernames
- Less profile customization
- Higher tweet frequencies
- Newer accounts

**Humans typically have:**

- More balanced follower ratios
- Personalized profiles
- Varied activity patterns
- Longer account histories
- Custom profile elements

## 📊 Visualizations

The project includes comprehensive visualizations:

- Distribution plots for bot vs human accounts
- Feature correlation heatmaps
- Model performance comparisons
- Feature importance rankings
- Cross-validation results

## 🛡️ Model Interpretability

- **Feature importance analysis** identifies key predictors
- **Correlation analysis** shows feature relationships
- **Statistical comparisons** between bot and human groups
- **Cross-validation** ensures model reliability

## 🚀 Future Improvements

- [ ] **Text Analysis**: Incorporate tweet content analysis
- [ ] **Network Analysis**: Add social network features
- [ ] **Temporal Analysis**: Include time-series behavior patterns
- [ ] **Ensemble Methods**: Combine multiple model types
- [ ] **Real-time Deployment**: API for live bot detection
- [ ] **Model Updating**: Continuous learning from new data

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### **Development Setup**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- **Dataset**: Thanks to [nahiar](https://huggingface.co/nahiar) for the Twitter bot detection dataset
- **Libraries**: Scikit-learn, XGBoost, LightGBM, and other amazing open-source tools
- **Community**: Machine learning and data science community for inspiration and resources

## 📚 References

1. [Twitter Bot Detection Research Papers]
2. [Machine Learning Best Practices]
3. [Feature Engineering Techniques]
4. [Model Evaluation Metrics]

---

⭐ **If you find this project helpful, please give it a star!** ⭐

---

_Last updated: July 2025_
