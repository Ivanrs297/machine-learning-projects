import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import dotenv_values

# Configuration of tracking server
# config = dotenv_values(".env")
config = dotenv_values(".env.testing")
mlfow_server_ip = config["MLFOW_SERVER_IP"]
mlflow.set_tracking_uri(mlfow_server_ip)


# Start experiment
name = "iris_logistic_model"
with mlflow.start_run(run_name = name):

    # load the iris dataset
    iris = load_iris()

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # create a logistic regression model
    lr = LogisticRegression(penalty="l2", max_iter=200)

    # train the model on the training data
    lr.fit(X_train, y_train)

    # make predictions on the testing data
    y_pred = lr.predict(X_test)

    # calculate the accuracy score of the model
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("penalty", "l2")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(lr, "model")
