# dsci-310-group-python2

## how to run docker files
docker build -t bill-analysis .

docker run -it --rm -v "${PWD}":/home/joyvan -p 8888:8888 bill-analysis