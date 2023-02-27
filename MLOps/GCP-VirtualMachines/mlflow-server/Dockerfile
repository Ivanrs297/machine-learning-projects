FROM nginx:1.17.6
# FROM ghcr.io/mlflow/mlflow:v2.0.1

# Remove default Nginx config
RUN rm /etc/nginx/nginx.conf
# Copy the modified Nginx conf
COPY nginx.conf /etc/nginx
# Copy proxy config
COPY mlflow.conf /etc/nginx/sites-enabled/


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install mlflow
CMD [ "python3","mlflow", "server", "-host", "0.0.0.0"  ]
