# IMDb Sentiment Analysis - MLOps Pipeline

[![CCDS](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

End-to-End Machine Learning (ML) project with an MLOps approach using **DVC**, **MLflow**, **Docker**, and **GitHub Actions**.

## **Project Overview**
This project performs sentiment analysis on IMDb movie reviews. The pipeline includes:
- Data extraction, ingestion, and cleaning.
- Feature engineering and model training.
- Model evaluation, versioning, and deployment as an API.
- Experiment tracking with MLflow and data versioning with DVC.
- Containerized workflows with Docker and automated CI/CD using GitHub Actions.

## **Project Organization**
```
├── LICENSE            <- Open-source license
├── README.md          <- Documentation of the project
├── docker-compose.yml <- Docker configuration for MLflow, API, and DVC pipeline
├── DVCPipelineDockerfile  <- Dockerfile for running DVC pipeline
├── MLflowDockerfile   <- Dockerfile for running MLflow server
├── apiDockerfile      <- Dockerfile for serving the model with FastAPI
│
├── api                <- API to serve the trained model
│   └── main.py        <- FastAPI script to load and serve the model
│
├── data               <- Stores datasets at different stages
│   ├── raw            <- Original dataset (immutable)
│   ├── clean          <- Processed dataset for training
│
├── models             <- Trained and registered models
│   ├── best_model.pkl <- Best-performing model
│   ├── model.pkl      <- Latest trained model
│
├── src                <- Source code for MLOps pipeline
│   ├── data_extraction.py         <- Fetches data
│   ├── data_ingestion_and_cleaning.py <- Prepares data
│   ├── feature_engineering.py     <- Creates model features
│   ├── model_training.py          <- Trains model
│   ├── model_evaluation.py        <- Evaluates model
│   ├── register_model.py          <- Registers model in MLflow
│   ├── load_model.py              <- Loads best model for inference
│
├── .github/workflows/cicd.yml  <- GitHub Actions CI/CD workflow
├── dvc.yaml           <- Defines DVC pipeline steps
├── dvc.lock           <- Tracks dataset versions
├── mlflow.db          <- MLflow SQLite database
├── mlruns/            <- MLflow experiment tracking folder
├── params.yaml        <- Stores hyperparameters and configurations
├── requirements.txt   <- Dependencies list
└── reports            <- Stores evaluation metrics and model performance reports
```

## **Pipeline DAG**
```
        +-----------------+        
        | data_extraction |        
        +-----------------+        
                  *
                  *
  +-----------------------------+  
  | data_ingestion_and_cleaning |  
  +-----------------------------+  
                  *
                  *
              +-------+
              | train |
              +-------+
           ***         ***
          *               *        
        **                 ***     
+----------+                  *    
| evaluate |               ***     
+----------+              *        
           ***         ***
              *       *
               **   **
         +----------------+        
         | register_model |        
         +----------------+        
+-----------------+
| load_best_model |
+-----------------+
```

## **Setup and Usage**
### **1. Setup Environment**
```sh
pip install -r requirements.txt
```

### **2. Data Versioning with DVC**
Initialize and track data with DVC:
```sh
dvc init
dvc add data/raw/dataset.zip
dvc push  # Push data to remote storage
```

To reproduce the pipeline:
```sh
dvc repro  # Runs all pipeline stages defined in dvc.yaml
```

### **3. Run MLflow for Experiment Tracking**
```sh
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### **4. Train Model and Log in MLflow**
```sh
python src/model_training.py
```

### **5. Evaluate Model**
```sh
python src/model_evaluation.py
```

### **6. Register Model**
```sh
python src/register_model.py
```

### **7. Serve Model with FastAPI**
```sh
uvicorn api.main:app --reload
```

## **CI/CD Workflow (GitHub Actions)**
This project uses **GitHub Actions** to automate testing, model training, and deployment.

### **Workflow Steps:**
1. **Code Linting & Formatting**
   - Ensures clean and standardized code formatting.
2. **Run Unit Tests**
   - Checks the correctness of individual components.
3. **DVC Data Validation**
   - Ensures dataset integrity before running the pipeline.
4. **Pipeline Execution**
   - Runs `dvc repro` to execute the complete ML pipeline.
5. **Model Validation & Registration**
   - Evaluates and registers the best-performing model.
6. **Deployment (Optional)**
   - Can be extended to deploy the model automatically.

### **Triggering CI/CD Pipeline**
The workflow runs on:
- **Push and pull requests** to `main` branch.
- **Manual trigger** via GitHub Actions UI.

## **Why We Use MLflow, DVC, Docker, and GitHub Actions?**
### **MLflow** (Experiment Tracking & Model Registry)
- Keeps track of experiments, hyperparameters, and metrics.
- Stores trained models and allows easy versioning.
- Enables model comparison and selection.
- Under DVC DAG, MLflow registers models and picks the best one based on accuracy.

### **DVC** (Data & Pipeline Versioning)
- Ensures reproducibility by tracking dataset versions.
- Automates the end-to-end pipeline execution under a single DVC DAG.
- Facilitates collaboration by managing large data files.

### **Docker** (Containerization & Deployment)
- Ensures consistent environments across development and deployment.
- Provides isolated runtime for MLflow, DVC, and API.
- Uses three Dockerfiles for DVC pipeline, MLflow, and API.
- Deploys everything with a single click using `docker-compose`.

### **GitHub Actions** (Automated CI/CD)
- Automates code testing and validation.
- Executes ML pipeline stages within a CI/CD environment.
- Ensures continuous integration of model updates.
- Simplifies deployment automation.

## **Conclusion**
This project showcases a complete **MLOps workflow** for sentiment analysis. Using **DVC-based DAG**, we ensure automation, reproducibility, and efficient model versioning with MLflow, all containerized with Docker for seamless deployment and continuously integrated using GitHub Actions.

---

💡 _Contributions are welcome! Feel free to fork and improve the project._

