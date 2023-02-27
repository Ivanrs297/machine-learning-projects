FROM nvcr.io/nvidia/pytorch:22.12-py3
WORKDIR /project
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY /project /project/
CMD ["echo", "All set!"]
