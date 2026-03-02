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
	test -f ./Resultados/metricas.txt

	echo "## Metricas del Modelo" > reporte.md
	cat ./Resultados/metricas.txt >> reporte.md

	echo '\n## Matriz de Confusion' >> reporte.md
	echo '![Matriz de Confusion](./Resultados/matriz_confusion.png)' >> reporte.md

	echo '\n## Curva ROC' >> reporte.md
	echo '![Curva ROC](./Resultados/roc_curve.png)' >> reporte.md

	echo '\n## SHAP - Importancia de Features' >> reporte.md
	echo '![SHAP](./Resultados/shap_summary.png)' >> reporte.md

	cml comment create reporte.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Actualizando los nuevos resultados"
	git push --force origin HEAD:update

configuracion_DVC_remoto:
	dvc remote add -f tumor_storage https://dagshub.com/alecorlo1234/ClasificadorTumorMama.dvc
	dvc remote default tumor_storage
	dvc remote modify tumor_storage auth basic
	dvc remote modify tumor_storage user alecorlo1234

hf-login:
	git fetch origin
	git checkout -B update
	git push -u origin update --force
	pip install -U "huggingface_hub[cli]"
	hf auth login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload alecorlo1234/ClasificadorTumorMama ./Aplicacion --repo-type=space --commit-message="Sincronizar archivos de Aplicacion"
	huggingface-cli upload alecorlo1234/ClasificadorTumorMama ./Modelo /Modelo --repo-type=space --commit-message="Sincronizar Modelo"

deploy: hf-login push-hub