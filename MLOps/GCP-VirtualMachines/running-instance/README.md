# NVIDIA-Docker on GCP GPU Virtual Machines
## Requirements
[NVIDIA Reference](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
1. Docker Engine
2. NVIDIA GPU Drivers
3. NVIDIA Container Toolkit [Installation Instructions]("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker")

## Installation
Building Docker Image

        $ cd docker
        $ sudo docker build --tag cv-inside .

Running a project
1. Verify the image **cv-inside** exists:

        $ sudo docker images

2. Create a container (<code>-itd</code> is required for bash entrypoint). By using <code>-v</code> argument the container will have a shared volume in <code>/container-results/</code> for saving the experiment results.

        $ sudo docker run -itd --name=cv-inside --runtime=nvidia --gpus all  -v ./container-results:/workspace/container-results --net cv-inside-net cv-inside bash

3. List containers

        $ sudo docker ps -a

4. If container is not running then:

        $ sudo docker start cv-inside

5. Access container bash
   
        $ sudo docker attach cv-inside

6. Inside the container, go to **src** folder. This folder must contain the source code of the project experiment.

        $ cd src/

7. Exit container without stopping it <code>Ctrl + p</code> then <code>Ctrl + q</code>


## Troubleshooting for VS Code Key Bindings
   1. Go to configuration and search **terminal.integrated.send**, and verify the checkbox is checked
   ![Demo SS](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/46759b3f-eceb-4943-8f5d-1cb4a5122ec3/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230227%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230227T014440Z&X-Amz-Expires=86400&X-Amz-Signature=419613ca0ac870af3b4a2bd6b25a8ad74b2c2dfd7f186a3b7fcb09994e469e08&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject "Configuration VS Code")


# Running experiments on MLFlow server
**Example** 
Create a environment file <code>example/.env</code>, with:

        MLFOW_SERVER_IP="http://127.0.0.1:5000"

Run <code>python examples/train_mflow.py</code>
