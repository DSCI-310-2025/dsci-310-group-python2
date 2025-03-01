# dsci-310-group-python2

## Steps to Run the Analysis:
First clone the repository: <br>
`git clone https://github.com/DSCI-310-2025/dsci-310-group-python2`

Run a git terminal and change the directory to the cloned repository: <br>
`cd dsci-310-group-python2`

Build a new docker instance: <br>
`docker build -t bill-analysis .`

Run the docker instance: <br>
`docker run -it --rm -v "${PWD}":/home/joyvan -p 8888:8888 bill-analysis`

Go to http://localhost:8888/ and make changes via the jupyter notebook