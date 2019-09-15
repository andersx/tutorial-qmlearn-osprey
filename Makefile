PIP=pip3
PYTHON=python3

all: pipeline run-cv-sklearn run-worker run-current-best

dependencies:

	#Install development QML
	${PIP} install git+https://github.com/qmlcode/qml@develop --user -U

	#Install Scikit-learn
	${PIP} install scikit-learn --user -U

	# Install Osprey
	${PIP} install sqlalchemy --user -U
	${PIP} install GPy --user -U
	${PIP} install git+git://github.com/larsbratholm/osprey.git --user -U


pipeline:
	${PYTHON} ./make_model_pickle.py

run-cv-sklearn:
	${PYTHON} ./qm7_cv_sklearn.py

run-worker:
	osprey worker qm7_fit.yaml

run-current-best:
	osprey current_best qm7_fit.yaml

clean:
	rm -f osprey-trials1.db 
	rm -f model.pickle
	rm -f idx.csv
