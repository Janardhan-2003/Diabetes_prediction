
# Prediction of Diabetes Using XGB Classifier

This project demonstrates predicting the likelihood of diabetes using various machine learning algorithms. The primary model is the XGB (Extreme Gradient Boosting) classifier, and comparisons are made with the K-Nearest Neighbors (KNN) and Support Vector Classifier (SVC) algorithms.

## Getting Started

### Installation

To run this project, install the necessary Python packages using the following commands:

```bash
pip install xgboost numpy pandas matplotlib sklearn seaborn
```

### Data

The dataset used is `diabetes.csv`, which should be placed in the `/content` directory. The dataset is from Kaggle and contains information on diabetic patients.

### Code Explanation

1. **Data Preprocessing**:
   - Load and inspect the dataset.
   - Check for missing values and visualize correlations.

2. **XGBClassifier**:
   - Train an XGB model on the training data.
   - Evaluate model performance using accuracy, confusion matrix, and classification report.
   - Predict outcomes for a sample input.

3. **Support Vector Classifier**:
   - Train an SVC model with a linear kernel.
   - Evaluate model performance using accuracy, confusion matrix, and classification report.
   - Predict outcomes for a sample input.

4. **K Nearest Neighbour**:
   - Train a KNN model with `n_neighbors=5`.
   - Evaluate model performance using accuracy, confusion matrix, and classification report.
   - Predict outcomes for a sample input.

### Running the Code

1. **Setup**: Ensure you have the dataset in the `/content` directory.
2. **Run**: Execute the script in a Jupyter Notebook or Python environment.

### Results

- **Accuracy Scores**: Provides the accuracy of each classifier.
- **Confusion Matrix**: Visualizes the performance of each model.
- **Classification Report**: Detailed metrics including precision, recall, and F1-score.

### Files

- `Pima_Indians_Diabetes.ipynb`: Jupyter Notebook with the complete code.
- `diabetes.csv`: Dataset used for training and testing.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgements

- The dataset is from Kaggle.
- Thanks to the developers of the XGBoost, Scikit-learn, and other libraries used in this project.

---

Feel free to modify any sections to better match your project's specifics!
