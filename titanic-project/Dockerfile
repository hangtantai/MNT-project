FROM python:latest
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy
RUN pip install xgboost
RUN pip install matplotlib
RUN pip install seaborn
WORKDIR  /src
COPY . /src/

CMD ["python","_Titanic_Project.py" ]
