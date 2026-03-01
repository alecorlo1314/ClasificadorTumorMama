install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black .

lint:
	pylint src/ entrenamiento.py --disable=R,C

train:
	python entrenamiento.py

eval:
	cml comment create Resultados/reporte.md

update-branch:
	git config --global user.name "github-actions"
	git config --global user.email "github-actions@github.com"
	git commit -am "Actualizando resultados del modelo"
	git push --force origin HEAD:update

configuracion_DVC_remoto:
	dvc remote add -f tumor_storage https://dagshub.com/alecorlo1234/ClasificadorTumorMama.dvc
	dvc remote default tumor_storage
	dvc remote modify tumor_storage auth basic
	dvc remote modify tumor_storage user alecorlo1234
	dvc remote modify tumor_storage password $(DAGSHUB_TOKEN)

hf-login:
	git fetch origin
	git switch -c update --track origin/update || git switch update
	pip install -U "huggingface_hub[cli]"
	git config --global credential.helper store
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload alecorlo1234/ClasificadorTumorMama ./Aplicacion --repo-type=space --commit-message="Sincronizar archivos de Aplicacion"
	huggingface-cli upload alecorlo1234/ClasificadorTumorMama ./Modelo /Modelo --repo-type=space --commit-message="Sincronizar Modelo"

deploy: hf-login push-hub