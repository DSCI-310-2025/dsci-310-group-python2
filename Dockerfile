### use python 3.11 with the jupyter notebook
### note: this docker library also installs jupyter 4.0.7
FROM jupyter/base-notebook:python-3.11

### give user permission to modify files in docker instance
USER root

### copy local files to jupyter instance
COPY . /home/jovyan

### Install system dependencies (including make)
RUN apt-get update && apt-get install -y \
    python3.11 \
    make \
    curl \
    python3-pip \
    gdebi-core \
    && rm -rf /var/lib/apt/lists/*
    
    
RUN curl -LO https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.42/quarto-1.6.42-linux-amd64.deb \
    && gdebi --non-interactive quarto-linux-amd64.deb \
    && rm -f quarto-linux-amd64.deb
    ### download libraries needed for analysis
RUN pip install pandas==2.2.3 \ matplotlib==3.10.1 \ seaborn==0.13.2 \ scikit-learn==1.6.1 \
    ucimlrepo==0.0.7 \ click==8.1.8 \ tabulate==0.9.0 && \
    ### skip token authentication needed for docker instance 
    echo "c.NotebookApp.token = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    chown -R jovyan /home/jovyan
