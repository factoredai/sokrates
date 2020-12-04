FROM python:3.7-slim-buster

LABEL NAME="Sokrates"\
      VERSION="0.2"\
      AUTHORS="Esteban Lopez, David Stiles"


WORKDIR /app

# Copy requirements file
COPY requirements.txt /app

# Install packages
RUN apt-get update\
    && pip install -r requirements.txt --no-cache-dir\
    && python -m nltk.downloader punkt

# Copy the rest of the files. Do this here to avoid reinstalling requirements
# if the python files change.
COPY ./ /app

# Expose port
EXPOSE 3000

# Start script
CMD bash start.sh