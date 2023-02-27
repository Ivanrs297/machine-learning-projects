
## Requierements 
[NVIDIA Reference](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
1. Docker Engine
2. NVIDIA GPU Drivers
3. NVIDIA Container Toolkit [Installation Instructions]("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker")


docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:xx.xx-py3

   
Building Docker Image

        $ sudo docker build --tag cv-inside .

Running a project
1. Verify the image **cv-inside** exists:

        $ sudo docker images

2. Create a container (<code>-itd</code> is required for bash entrypoint)

        $ sudo docker run -itd --name=<container_name> --runtime=nvidia --gpus all <image_id> bash

3. List containers

        $ sudo docker ps -a

4. If container is not running then:

        $ sudo docker start <container_name>

5. Access container bash
   
        $ sudo docker attach <container_name>

6. Exit container without stopping it <code>Ctrl + p</code> then <code>Ctrl + q</code>
   1. Troubleshooting for VS Code
   2. Go to configuration and search **terminal.integrated-send**, and verify the checkbox is checked
   3. ![Demo SS](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/46759b3f-eceb-4943-8f5d-1cb4a5122ec3/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230227%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230227T014440Z&X-Amz-Expires=86400&X-Amz-Signature=419613ca0ac870af3b4a2bd6b25a8ad74b2c2dfd7f186a3b7fcb09994e469e08&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject "Configuration VS Code")
   4. 
