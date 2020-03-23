# AI Simulator

## Project Organization

* `./doc/`: The detailed document of the AI Simulator.
* `./src/`: The source code of the AI Simulator.
* `./tests/`: The unit test files used in the AI Simulator.
* `./Dockerfile`: The Dockerfile used to build the docker environment for the AI Simulator.
* `./requirements.txt`: The prerequisite Python packages for the AI Simulator.
* `./__init__.py`: Blank file for Python project.
* `./READEME.md`: The README file.

## Installation

### Install by Dockerfile (Recommended)

Using AI Simulator at Docker environment is the easiest and recommended method.
```
# Build AI Simulator
sudo docker build -t ai-simulator .

# Run AI Simulator experiment
sudo docker run ai-simulator
```

### Install on Native Machine

Please install Python and TensorFlow berfore running the AI Simultor. The recommended setting is as follows:

```text
    Python: 3.5–3.7 (64 bit)
    Pip: >= 19.0
    TensorFlow: 1.15
    Protobuf: 3.8
    Pytest: 5.3.2
```
- Python

    Because TensorFlow on Windows only supports Python 3.x (64-bit), we highly recommend to insatll Python 3.x (64-bit). You can download and install Python from [Python Website](https://www.python.org/). You can update the pip to latest version.

    ```python
    # Install python on Ubuntu
    apt-get -y install python3-dev python3-pip

    # Install pip3
    python3 -m pip install --upgrade pip

    # Check your version
    python3 --version
    pip3 --version
    ```

- Python Virtual Environment (Optional, Recommended)

    Python virtual environments are used to isolate package installation from the system.

    ```python
    # Install virtualenv
    pip3 install -U pip virtualenv
    virtualenv --version

    # Create a new virtual environment under .\venv directory
    virtualenv --system-site-packages -p python3 ./venv

    # Activate the environment
    # For Windows
    .\venv\Scripts\activate
    # For Ubuntu
    source ./venv/bin/activate

    # Deactivate the environment
    deactivate
    ```

- TensorFlow, Protobuf, and Pytest
    Goto the the ai_simulator folder, and run:

    ```python
    # Install TensorFlow, Protobuf, and Pytest
    python3 -m pip install -r requirements.txt

    # Verify the installation
    python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    python3 -m pytest -v
    ```

## Run Test

You can go to ai_simulator folder, and run test:

```python
python3 -m pytest -v
```