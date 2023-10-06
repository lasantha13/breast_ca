import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "breast_ca"


list_of_files = [
    ".github/workflows/.gitkeep",
    f"notebook/test.ipynb",
    f"notebook/data/data.csv",
    f"src/__init__.py",
    f"src/exception.py",
    f"src/logger.py",
    f"src/utils.py",
    f"src/components/__init__.py",
    f"src/components/data_ingestion.py",
    f"src/components/data_ingestion_trasformation.py",
    f"src/pipeline/__init__.py",
    f"src/pipeline/predict_pipeline.py",
    f"src/pipeline/train_pipeline.py",
    "app.py",
    "requirements.txt",
    "setup.py"
    


]




for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")