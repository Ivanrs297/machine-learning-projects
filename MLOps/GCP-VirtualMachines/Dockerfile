FROM nvcr.io/nvidia/pytorch:22.12-py3
WORKDIR /project
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY /project /project/
CMD ["echo", "All set!"]
