ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

ARG PROJECT_NAME=universalfakedetect
ARG USER_NAME=challenger
ARG GROUP_NAME=challengers
ARG UID=1000
ARG GID=1000
ARG PYTHON_VERSION=3.8
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}
ARG RUN_POETRY_INSTALL_AT_BUILD_TIME="false"
# Followings are for submodules
ARG HYPERINVERTER_ROOT_DIRECTORY=${APPLICATION_DIRECTORY}/submodules/HyperInverter
ARG CLEVERHANS_ROOT_DIRECTORY=${APPLICATION_DIRECTORY}/submodules/Cleverhans

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${APPLICATION_DIRECTORY}

# build-essential, python3-dev are needed for "insightface".
# libopencv-dev is needed for "cv2".
RUN apt-get update -y

RUN apt-get install --no-install-recommends -y software-properties-common
RUN apt-get install --no-install-recommends -y git
RUN apt-get install --no-install-recommends -y curl
RUN apt-get install --no-install-recommends -y make
RUN apt-get install --no-install-recommends -y ssh
RUN apt-get install --no-install-recommends -y openssh-client
RUN apt-get install --no-install-recommends -y screen
RUN apt-get install --no-install-recommends -y build-essential
RUN apt-get install --no-install-recommends -y python3-dev
RUN apt-get install --no-install-recommends -y libopencv-dev
RUN apt-get install --no-install-recommends -y python${PYTHON_VERSION} python3-pip python-is-python3

# RUN apt update && apt install --no-install-recommends -y software-properties-common \
#     git curl make ssh openssh-client screen build-essential python3-dev libopencv-dev \
#     python${PYTHON_VERSION} python3-pip python-is-python3

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
