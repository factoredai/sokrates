FROM python:3.7-slim-buster

LABEL NAME="Sokrates"\
      VERSION="0.1"\
      AUTHORS="Esteban Lopez, David Stiles"


WORKDIR /app

# Copy files
COPY ./ /app

# Install packages
RUN apt-get update\
    && pip install -r requirements.txt --no-cache-dir\
    && python -m nltk.downloader punkt

# Expose port
EXPOSE 3000

CMD bash start.sh