# This repo is one of my templates.

In a few words this project is a neural network with api and its chip deploy to a vps via ssh.

It contains the following things:
1. hydra configs
2. lightning scripts to train a neural network
3. Converting a torch nn to onnx
4. FastAPI
5. Dockerfile
6. Makefile

Deploying process assumes you have a vps with configured ssh keys and docker installed.

You can see example of the deployed app by following the link bellow:
https://minecraftskins.webtm.ru/download_skin
If you already clicked that link then you downloaded a minecraft skin generated from gaussian noise by very very small WGAN-GP. You can see it 3D on some website like that https://customuse.com/free-tools/minecraft-clothes-preview.

Usage:
1. git clone
2. configure poetry from zero by deleting pyproject.toml and poetry.lock files
3. download some data
4. correct Lightning (Data) Modules and Hydra configs
5. correct onnx converting script
6. correct app
7. correct Dockerfile
8. optionally add k8s
9. correct Makefile (or replace it with jenkins, gitlab, ....)

TODO:
k8s
gitlab
