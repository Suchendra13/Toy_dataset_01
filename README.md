# Iris Flower Classification using K-Nearest Neighbors (KNN)

This project demonstrates a simple machine learning classification model to predict the species of an Iris flower based on its sepal and petal measurements. The model is built using the K-Nearest Neighbors (KNN) algorithm.

## Dataset

The project uses the classic **Iris dataset**, available in `scikit-learn`. The dataset contains 150 samples of Iris flowers, each with four features:
* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

The target variable is the species of the flower, which belongs to one of three classes:
1.  Setosa
2.  Versicolor
3.  Virginica

## Project Workflow

The Jupyter Notebook `Toy_dataset_01.ipynb` covers the following steps:
1.  **Data Loading:** The Iris dataset is loaded from `sklearn.datasets`.
2.  **Exploratory Data Analysis (EDA):** Basic information and statistics of the dataset are explored using `pandas`. A `pairplot` is generated with `seaborn` to visualize the relationships between features for each species.
3.  **Data Preparation:** The data is split into features (X) and the target variable (y).
4.  **Train-Test Split:** The dataset is divided into an 80% training set and a 20% testing set.
5.  **Model Training:** A K-Nearest Neighbors classifier (with `n_neighbors=3`) is trained on the training data.
6.  **Prediction & Evaluation:** The trained model is used to make predictions on the test set, and its accuracy is calculated.

## Results

The KNN model achieved an accuracy of **100%** on the test set for this dataset. The notebook also includes an example of how to predict the species for a new flower with given measurements.

## How to Run

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Suchendra13/Toy_dataset_01.git](https://github.com/Suchendra13/Toy_dataset_01.git)
    ```
2.  **Install the required libraries:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
3.  **Run the Jupyter Notebook:**
    Open and run the `Toy_dataset_01.ipynb` file in a Jupyter environment.