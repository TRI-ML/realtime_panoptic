# Makefile for launching common tasks

PYTHON ?= python
DOCKER_OPTS ?= \
    -v /dev/shm:/dev/shm \
	-v /root/.ssh:/root/.ssh \
	-v /var/run/docker.sock:/var/run/docker.sock \
	--network=host \
	--privileged
PACKAGE_NAME ?= panoptic
WORKSPACE ?= /workspace/$(PACKAGE_NAME)
DOCKER_IMAGE_NAME ?= $(PACKAGE_NAME)
DOCKER_IMAGE ?= $(DOCKER_IMAGE_NAME):latest

all: clean test

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

clean-logs:
	find . -name "tensorboardx" | xargs rm -rf && \
	find . -name "wandb" | xargs rm -rf

test:
	PYTHONPATH=${PWD}/tests:${PYTHONPATH} python -m unittest discover -s tests

docker-build:
	docker build \
		-f docker/Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-run-test-sample:
	nvidia-docker run --name panoptic --rm \
		-e DISPLAY=${DISPLAY} \
		-v ${PWD}:${WORKSPACE} \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ~/.torch:/root/.torch \
		-p 8888:8888 \
		-p 6006:6006 \
		-p 5000:5000 \
		-it \
		-v ${PWD}:${WORKSPACE} \
		${DOCKER_OPTS} \
		${DOCKER_IMAGE} bash -c \
		"wget -P /workspace/panoptic/ -c https://tri-ml-public.s3.amazonaws.com/github/realtime_panoptic/models/cvpr_realtime_pano_cityscapes_standalone_no_prefix.pth && \
		python scripts/demo.py \
		--config-file configs/demo_config.yaml \
		--input media/figs/test.png \
		--pretrained-weight cvpr_realtime_pano_cityscapes_standalone_no_prefix.pth"

docker-start:
	nvidia-docker run --name panoptic --rm \
		-e DISPLAY=${DISPLAY} \
		-v ${PWD}:${WORKSPACE} \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /data:/data \
		-v ~/.torch:/root/.torch \
		-p 8888:8888 \
		-p 6006:6006 \
		-p 5000:5000 \
		-d \
		-it \
		${DOCKER_OPTS} \
		${DOCKER_IMAGE} && \
		nvidia-docker exec -it panoptic bash
