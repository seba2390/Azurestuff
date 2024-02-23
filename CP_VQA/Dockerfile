FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc libpython3-dev cmake libboost-all-dev

# Upgrade pip and install wheel
RUN pip install --upgrade pip
RUN pip install wheel

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY . /app/

# Run tests
# ignore::DeprecationWarning -W ignore::PendingDeprecationWarning is to ignore Qiskit warnings
RUN find ./test -name '*.py' -exec pytest {} -W ignore::DeprecationWarning -W ignore::PendingDeprecationWarning \;

# Command to run the application when the container starts
CMD ["python", "main.py"]
