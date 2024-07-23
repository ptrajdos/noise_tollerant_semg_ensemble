ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))
DATAFILE=${ROOTDIR}/tsnre_windowed.tar.xz
DATAFILEID=15D-PUeOIHQXyJMhaoDGOxmSfAayMyBgb
DATADIR=${ROOTDIR}/data
VENV_SUBDIR=${ROOTDIR}/venv
EXPERIMENT_SUBDIR=${ROOTDIR}/dexterous_bioprosthesis_2021_raw_datasets_framework_experiments

PYTHON=python
PIP=pip
CURL=curl
TAR=tar

.PHONY: all clean

.NOTPARALLEL: run

create_env: venv data

clean:
	rm -rf ${VENV_SUBDIR}

run_one_class: create_env
	. ${VENV_SUBDIR}/bin/activate; ${PYTHON} ${EXPERIMENT_SUBDIR}/outlier_detection_experiment_snr2.py
run_commitees: create_env
	. ${VENV_SUBDIR}/bin/activate; ${PYTHON} ${EXPERIMENT_SUBDIR}/dexterous_bioprosthesis_2021_raw_datasets_framework_experiments/channel_combination_ensemble_fast_full.py

run_reference: create_env
	. ${VENV_SUBDIR}/bin/activate; ${PYTHON} ${EXPERIMENT_SUBDIR}/dexterous_bioprosthesis_2021_raw_datasets_framework_experiments/channel_combination_ensemble_full_2.py

run: run_one_class run_commitees run_reference
	echo "Experiments performed"

venv:
	${PYTHON} -m venv ${VENV_SUBDIR}
	. ${VENV_SUBDIR}/bin/activate; ${PIP} install -e .

data:
	mkdir -p ${DATADIR}
	${CURL} -L -o ${DATAFILE} "https://drive.usercontent.google.com/download?id=${DATAFILEID}&export=download&authuser=1&confirm=t"
	${TAR} -xvf ${DATAFILE} --directory ${DATADIR}
	
