import os
from pathlib import Path
import sys

list_of_files = [
    ".github",
    "experiments/experiment1.ipynb",
    "experiments/experiment2.ipynb",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/base_model_dev.py",
    "src/components/model_training.py",
    "src/components/model_evaluation.py",
    "src/constants/__init__.py",
    "src/utils/utils.py",
    "src/utils/__init__.py",
    "src/loggingInfo/__init__.py",
    "src/loggingInfo/loggingInfo.py",
    "src/exception/__init__.py",
    "src/exception/exception.py",
    "src/pipeline/__init__.py",
    "src/config/entity.py",
    "src/config/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/pipeline/model_pipeline.py",
    "config/config.yaml",
    "params.yaml",
    ".gitignore",
    "Dockerfile",
    "dvc.yaml",
    "docker_compose.yaml",
    "app.py",
    "main.py",
    "requirements.txt",
    "setup.py",
    ".env"
]

for file in list_of_files:
    filePath = Path(file)
    fileDir, fileExtPath = os.path.split(file)
    
    if (fileDir != ""):
        os.makedirs(fileDir, exist_ok=True)
        
    if (not os.path.exists(filePath)):
        with open(filePath, "w") as f:
            pass