# Installation

Activate firewall

        sudo ufw enable
        sudo ufw allow OpenSSH
        sudo ufw allow http
        sudo ufw allow https
        sudo ufw allow 5000
        sudo ufw status

Build the image

        sudo docker build --tag cv-inside-tracking .

Run the image

        sudo docker run -itd --name=cv-inside-tracking -p 5000:80 cv-inside-tracking bash

On the container

        mlflow server --host 0.0.0.0


Create a <code>.env</code> file, example:

        MLFOW_SERVER_IP="http://127.0.0.1:5000"