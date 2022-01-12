SHELL=/bin/bash
POETRY_VERSION=1.1.10
CUML_HOME=./cuml

POETRY = pip3 install -q poetry==${POETRY_VERSION}

CUML = git clone https://github.com/rapidsai/cuml.git ${CUML_HOME} \
		&& cd ${CUML_HOME}/cpp \
		&& mkdir -p build && cd build \
		&& export CUDA_BIN_PATH=${CUDA_HOME} \
		&& cmake .. \
		&& make install \
		&& cd ../../python \
		&& python setup.py build_ext --inplace \
		&& python setup.py install


poetry:
	${POETRY}

develop: # usually use this command
	${POETRY} \
	&& poetry install \
	&& poe force-cuda11


develop_no_venv:
	${POETRY} \
	&& poetry config virtualenvs.create false --local \
	&& poetry install \
	&& poe force-cuda11

set_tpu:
	${POETRY} \
	&& poetry config virtualenvs.create false --local \
	&& poetry install \
	&& poetry run python3 -m pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl \

cuml:
	${CUML}

pip_export:
	pip3 freeze > requirements.txt

poetry_export:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

develop_by_requirements:
	for package in $(cat requirements.txt); do poetry add "${package}"; done

update_datasets:
	zip -r output/sub.zip output/sub
	kaggle datasets version -p ./output/sub -m "Updated data"

pull_kaggle_image:
	docker pull gcr.io/kaggle-gpu-images/python
