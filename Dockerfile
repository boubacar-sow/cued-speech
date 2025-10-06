# Dockerfile - test cued-speech + pixi (Python 3.11.5)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
# installer outils système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential unzip ffmpeg wget \
  && rm -rf /var/lib/apt/lists/*

# installer pixi (script officiel)
RUN curl -fsSL https://pixi.sh/install.sh | bash

# rendre pixi disponible dans PATH pour toutes les commandes Docker RUN
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /workspace
# créer un espace de travail dune manière reproducible
RUN mkdir /workspace/cued-speech-env
WORKDIR /workspace/cued-speech-env

# initialiser pixi workspace
RUN pixi init
RUN pixi add "python==3.11"

# ajouter montreal-forced-aligner (MFA) recommandé par la doc
# (version prise de l'exemple sur PyPI; change si nécessaire)
RUN pixi add montreal-forced-aligner=3.3.4

# ajouter openai-whisper depuis PyPI dans l'environnement pixi
RUN pixi add --pypi openai-whisper

# installer l'environnement (résolution + installation)
RUN pixi install

# pré-télécharger et mettre en cache le modèle Whisper medium pendant la build
# ceci évite de le télécharger pendant la génération cued-speech
RUN pixi run python -c "import whisper; whisper.load_model('medium')"

# installer le package cued-speech via pip dans l'environnement pixi
# (pixi run exécute la commande dans l'environnement)
RUN pixi run python -m pip install --upgrade pip setuptools wheel
RUN pixi run python -m pip install cued-speech


