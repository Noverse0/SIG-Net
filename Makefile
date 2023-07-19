default: build

help:
	@echo 'Management commands for sig-net:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the sig-net project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t sig-net 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name sig-net -v `pwd`:/workspace/sig-net sig-net:latest /bin/bash

up: build run

rm: 
	@docker rm sig-net

stop:
	@docker stop sig-net

reset: stop rm
