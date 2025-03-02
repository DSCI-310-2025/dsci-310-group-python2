### use python 3.11 with the jupyter notebook
### note: this docker library also installs jupyter 4.0.7
FROM jupyter/base-notebook:python-3.11

### download libraries to use for analysis
RUN pip install pandas==2.2.3
RUN pip install matplotlib==3.10.1
RUN pip install seaborn==0.13.2
RUN pip install scikit-learn==1.6.1

### copy local files to jupyter instance
COPY . /home/jovyan/

### skip token authentication needed for docker instance 
RUN mkdir -p /home/jovyan/.jupyter && \
    echo "c.NotebookApp.token = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py

### give user permission to modify files in docker instance
USER root

RUN mkdir -p /home/jovyan && \
    chown -R jovyan /home/jovyan