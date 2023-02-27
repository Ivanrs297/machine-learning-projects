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

Run the image

        sudo docker run -itd --name=cv-inside-tracking -p 5000:5000 --net tulip-net cv-inside-tracking bash

Attach container

        sudo docker attach cv-inside-tracking

On the container

        mlflow server -p 80 --host 0.0.0.0 

Or

        sudo docker compose up -d


Create a <code>.env</code> file, example:

        MLFOW_SERVER_IP="http://127.0.0.1:5000"