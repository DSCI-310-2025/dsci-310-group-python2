### use python 3.11 with the jupyter notebook
### note: this docker library also installs jupyter 4.0.7
FROM jupyter/base-notebook:python-3.11

### copy local files to jupyter instance
COPY . /home/jovyan/

### give user permission to modify files in docker instance
USER root

### download libraries to use for analysis
RUN pip install pandas==2.2.3 && \
    pip install matplotlib==3.10.1 && \
    pip install seaborn==0.13.2 && \
    pip install scikit-learn==1.6.1 && \
    pip install ucimlrepo==0.0.7 && \
    pip install click==8.1.8 && \
    ### skip token authentication needed for docker instance 
    echo "c.NotebookApp.token = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    chown -R jovyan /home/jovyan
