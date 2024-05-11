ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

ARG PROJECT_NAME=universalfakedetect
ARG USER_NAME=challenger
ARG GROUP_NAME=challengers
ARG UID=1000
ARG GID=1000
ARG PYTHON_VERSION
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}
ARG RUN_POETRY_INSTALL_AT_BUILD_TIME="false"

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${APPLICATION_DIRECTORY}

RUN apt update -y && apt install --no-install-recommends -y software-properties-common \
    git curl make ssh openssh-client screen build-essential python3-dev libopencv-dev \
    python3-pip python-is-python3 

# deadsnakes PPAの追加とPython 3.9のインストール
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update -y && \
    apt install --no-install-recommends -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils && \
    python${PYTHON_VERSION} -m pip install --upgrade pip

# update-alternativesを使ってデフォルトのPythonをPython 3.9に設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

RUN python3 -m pip install --upgrade pip setuptools requests \
    && python3 -m pip install poetry

# Add user. Without this, following process is executed as admin.
RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}

USER ${USER_NAME}
WORKDIR ${APPLICATION_DIRECTORY}

# If ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true", install Python package by Poetry and move .venv under ${HOME}.
# This process is for CI (GitHub Actions). To prevent overwrite by volume of docker compose, .venv is moved under ${HOME}.
COPY --chown=${UID}:${GID} pyproject.toml poetry.lock poetry.toml .
RUN test ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true" && poetry install || echo "skip to run poetry install."
RUN test ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true" && mv ${APPLICATION_DIRECTORY}/.venv ${HOME}/.venv || echo "skip to move .venv."
