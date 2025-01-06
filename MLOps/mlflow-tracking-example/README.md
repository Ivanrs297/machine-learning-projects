# mlops-mlflow-tracking

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MICAI 2024 Tutorial for MLOps for Medical Imaging Made Easy

## Prerequisites

- **Python 3.6+** installed on your machine.
- Basic understanding of Python, PyTorch, and machine learning concepts.
- Install the required libraries:

```bash
pip install mlflow medmnist torch torchvision matplotlib seaborn
```


## MLFlow

Create a folder `./data/raw/` within the root of the project.

Run the following command in your terminal to start the MLFlow UI:

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view your experiment.


### Tracking system metrics

To track system metrics like CPU usage, memory consumption, GPU usage, and disk I/O in MLFlow, we can use the psutil library for system metrics and GPUtil for GPU monitoring (if using a GPU). These metrics can be logged as part of the MLFlow experiment to provide a comprehensive view of the resource usage during model training and evaluation.

```bash
pip install psutil gputil pynvml
```

## Serve the Model

The model URI will have the form: runs:/<RUN_ID>/MNIST_CNN_Model, where <RUN_ID> is the ID of the run where the model was logged.

```bash
mlflow models serve -m runs:/<RUN_ID>/PathMNIST_cnn_model --no-conda
```
Replace <RUN_ID> with your actual run ID.

This will start an HTTP server on port 5000 (by default), exposing the model as a REST API endpoint at http://127.0.0.1:5000/invocations.

Example Command:

```bash
mlflow models serve -m runs:/adca5b681fb54667a3b5e82114034130/PathMNIST_cnn_model --no-conda -p 5001
``` 

### Script: `serve_model.sh`

```bash
#!/bin/bash

# Set the MLFlow tracking URI
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Check if RUN_ID is provided as an argument
if [ -z "$1" ]; then
    echo "Error: You must provide the RUN_ID as an argument."
    echo "Usage: ./serve_model.sh <RUN_ID>"
    exit 1
fi

# Store the provided RUN_ID
RUN_ID=$1

# Serve the model using the provided RUN_ID
mlflow models serve -m runs:/$RUN_ID/MNIST_CNN_Model --no-conda -p 5001
```

### How to Use the Script:

1. **Save the script** as `serve_model.sh`.
2. **Make it executable** by running:

   ```bash
   chmod +x serve_model.sh
   ```

3. **Run the script** by providing the `RUN_ID` as an argument:

   ```bash
   ./serve_model.sh <RUN_ID>
   ```

   Replace `<RUN_ID>` with the actual run ID from your **MLFlow** run. The script will:
   - Set the `MLFLOW_TRACKING_URI` to point to the **MLFlow** tracking server (`http://127.0.0.1:5000`).
   - Serve the model associated with the provided `RUN_ID`.

### Example:

```bash
./serve_model.sh 208ba84905d341e09932a54a74a82fee
```



## DVC (Data Version Control)

### 1. Install DVC and Initialize It


If you haven't already installed **DVC**, do so by running:

```bash
pip install dvc
```

Next, initialize DVC in your project:

```bash
dvc init
```

### 2. Configure DVC to Use Local Storage

We'll configure DVC to use a local directory as remote storage. You can use any directory on your system to store DVC-tracked files. For instance, you can create a directory called `dvc-storage` in your project folder or elsewhere on your system.

```bash
mkdir dvc-storage
```

Configure DVC to use this directory as a local remote:

```bash
dvc remote add -d localremote ./dvc-storage
```

### 3. Track the Dataset with DVC

Now, let's track the dataset using DVC. Assuming that the PathMNIST dataset is in a `data/` directory, run the following commands:

```bash
# Track the dataset with DVC
dvc add data/
```

This command will create a `.dvc` file (`data.dvc`) to track the `data/` directory.

You can commit this to Git to keep track of the changes:

```bash
git add data.dvc .gitignore
git commit -m "Add PathMNIST dataset to DVC tracking"
```

Push the dataset to the local DVC storage:

```bash
dvc push
```

This will move the dataset to the local storage (`dvc-storage`).

### 4. Track the Model Checkpoints with DVC

After training the model, you can track the model files as well. For example, let's say your model checkpoints are saved in a `models/` directory.

To track the model:

```bash
dvc add models/
```

Just like with the dataset, this command will create a `.dvc` file (`models.dvc`) to track the `models/` directory.

Commit the DVC tracking for the models to Git:

```bash
git add models.dvc .gitignore
git commit -m "Track models with DVC"
```

Push the models to the local DVC storage:

```bash
dvc push
```

### 5. Pull the Dataset and Model Checkpoints Locally

If you want to reproduce the experiment or share the repository with others, they can easily retrieve the dataset and models from local storage using DVC.

To pull the dataset and models:

```bash
dvc pull
```

This will fetch the dataset and model checkpoints from the local storage (`dvc-storage`).
