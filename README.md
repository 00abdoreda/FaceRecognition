# FaceRecognition

## Install

1. Clone or copy the project.
2. Create and activate a virtual environment:
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Train Model

1. Run the training script:
    ```
    python training/train.py
    ```
2. Confirm the model artifacts are saved in `model/`.

## Run on 0.0.0.0

1. Start the application:
    ```
    ```
    cd api/
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    ```
2. Open the service at:
    ```
    http://0.0.0.0:8000
    ```

Adjust file names and ports if your project uses different scripts or configuration.