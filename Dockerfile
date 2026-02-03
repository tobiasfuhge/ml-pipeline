FROM bitnamilegacy/spark:4.0.0-debian-12-r20

USER root

RUN pip install \
    mlflow \
    boto3

COPY eda_spark.py /opt/spark/app/
COPY preprocessing_spark.py /opt/spark/app/
COPY train_spark.py /opt/spark/app/
COPY evaluation_spark.py /opt/spark/app/

USER 1001
