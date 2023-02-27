# Installation

Activate firewall on VM (not docker)

        sudo ufw enable
        sudo ufw allow OpenSSH
        sudo ufw allow http
        sudo ufw allow https
        sudo ufw allow 5000
        sudo ufw status

Build the image

        sudo docker build --tag cv-inside-tracking .

Create a bridge network

        sudo docker network create cv-inside-net

Run the image (Important: hostname and network arguments)

        sudo docker run -itd --name cv-inside-tracking -h=cv-inside-tracker -p 5000:5000 --net cv-inside-net cv-inside-tracking bash

Attach container

        sudo docker attach cv-inside-tracking

On the container

        mlflow server --host 0.0.0.0 

Or docker compose (TODO)

        sudo docker compose up -d

TODO: NGINX Integration