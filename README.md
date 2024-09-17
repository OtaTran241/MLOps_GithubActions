# MLOps Model Deployment

this MLOps project is an end-to-end pipeline for a Machine Learning model using GitHub Actions.

## Customer Churn Prediction Model

This repository contains code for training a machine learning model to predict customer churn using a dataset of bank customers. The goal is to identify customers who are likely to churn, allowing for targeted retention strategies.

The `train.py` script performs the following tasks:

1. **Data Preparation**: Loads the dataset, handles missing values, and encodes categorical features.
2. **Feature and Target Variable Extraction**: Separates features from the target variable.
3. **Data Splitting**: Divides the data into training and testing sets.
4. **Model Selection and Hyperparameter Tuning**: Evaluates multiple classifiers with different hyperparameters using grid search to optimize for recall.  
   -Recall was selected as the primary evaluation metric for our model because it is crucial for detecting all relevant instances of churn, especially in an imbalanced dataset. By optimizing recall, we ensure that we identify as many at-risk customers as possible, which helps in effective customer retention and minimizes the impact of churn on the business.
5. **Model Training**: Trains the best-performing model based on recall.
6. **Evaluation**: Assesses model performance using various metrics and saves evaluation results and visualizations.

## CI/CD Pipeline Analysis

The CI/CD pipeline defined in `.github/workflows/CI_CD.yml` automates the process of training, testing, building, and deploying the machine learning model. Here’s a breakdown of the key components:

### 1. **Trigger**
- The pipeline is triggered on every push to the repository.

### 2. **Jobs**
The pipeline consists of three main jobs:

#### a. **train_and_test**
- **Environment**: Runs on `ubuntu-latest` using a Docker container.
- **Steps**:
  - **Checkout Code**: Checks out the repository code.
  - **Install Dependencies**: Installs required Python packages and CML.
  - **Train Model and Log Results**: 
    - Trains the model using various classifiers and scalers.
    - Selects the best model based on recall score.
    - Saves the trained model to `models/model.pkl`.
    - Logs metrics, including training and test scores, best recall score, and best threshold, to `metrics.txt`.
    - Generates and saves confusion matrix and classification report as images.
  - **Run Tests**:
    - Executes unit tests to ensure the functionality of the code.
    - Logs the results of single test cases to `single_test.txt`.
  - **Report**: Uses CML to comment on the training report.

#### b. **build**
- **Dependencies**: This job depends on the successful completion of the `train_and_test` job.
- **Steps**:
  - **Checkout code**: Retrieves the latest code.
  - **Log in to Docker Hub**: Authenticates to Docker Hub using secrets.
  - **Build Docker image**: Builds a Docker image for the application.
  - **Push Docker image**: Pushes the built image to Docker Hub.

#### c. **deploy**
- **Dependencies**: This job depends on the successful completion of the `build` job.
- **Steps**:
  - **Checkout code**: Retrieves the latest code.
  - **Install Kubectl**: Installs the Kubernetes command-line tool.
  - **Configure AWS credentials**: Sets up AWS credentials for accessing EKS.
  - **Update kubeconfig for EKS**: Configures access to the EKS cluster.
  - **Replace Docker Hub username in deployment.yml**: Updates the deployment configuration with the Docker Hub username.
  - **Deploy to Kubernetes**: Applies the Kubernetes deployment configuration.
  - **Verify deployment**: Checks the status of the deployment to ensure it is running correctly.

### Conclusion
This CI/CD pipeline streamlines the process of deploying machine learning models, ensuring that code changes are automatically tested, built, and deployed to a Kubernetes environment.
