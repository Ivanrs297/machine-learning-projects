@echo off

:: Set the MLFlow tracking URI
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000

:: Check if RUN_ID is provided as an argument
if "%1" == "" (
    echo Error: You must provide the RUN_ID as an argument.
    echo Usage: serve_model.bat ^<RUN_ID^>
    exit /b 1
)

:: Store the provided RUN_ID
set RUN_ID=%1

:: Serve the model using the provided RUN_ID
mlflow models serve -m runs:/%RUN_ID%/MNIST_CNN_Model --no-conda