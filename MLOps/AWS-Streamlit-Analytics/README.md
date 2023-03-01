1. Create an EC2 instance [Here](https://aws.amazon.com/es/ec2/)
2. On Terminal
   
        sudo su
        yum update
        yum install git
        yum install python3-pip
        python3 -m pip install -r requirements.txt
        python3 -m streamlit run app.py --server.port 80

3. Run app continuously

        nohup python3 -m streamlit run app.py --server.port 80