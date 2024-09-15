# MLOps Model Deployment

This project demonstrates the deployment of a machine learning model using Flask, Docker, and Kubernetes. The model is trained on the Iris dataset and exposes a REST API for predictions.

## CI/CD Pipeline Analysis

The CI/CD pipeline defined in `.github/workflows/cd.yml` automates the process of training, testing, building, and deploying the machine learning model. Here’s a breakdown of the key components:

### 1. **Trigger**
- The pipeline is triggered on every push to the repository.

### 2. **Jobs**
The pipeline consists of three main jobs:

#### a. **train_and_test**
- **Environment**: Runs on `ubuntu-latest` using a Docker container.
- **Steps**:
  - **Checkout code**: Retrieves the latest code from the repository.
  - **Install dependencies**: Installs Python and Node.js dependencies.
  - **Train Model and Log Results**: Executes the training script and logs the results into a report.
  - **Run tests**: Executes the test script and appends results to the report.
  - **Report**: Uses CML to comment on the report in the pull request.

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
