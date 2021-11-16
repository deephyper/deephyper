FROM continuumio/miniconda3

WORKDIR /app

# RUN apt-get update -y
# RUN apt-get install gcc g++ cmake vim -y
# RUN apt-get install -y build-essential

RUN conda create -n dh python=3.9 -y --quiet

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "dh", "/bin/bash", "-c"]

# copy the repo
COPY . .

# install the package
RUN conda install gxx_linux-64 gcc_linux-64
RUN pip install -e '.[analytics]'


# activate 'pomdp' environment by default
RUN echo "conda activate dh" >> ~/.bashrc

# start in /app/exp
WORKDIR /app/exp