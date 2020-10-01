# Informe Trabajo Parcial Machine Learning

## Integrantes:
- Alegre Flores, Renzo Paolo
- Baldeón Albornoz, Braulio Sebastián
- Nuñez Robinson, Daniel

## Introducción

El presente trabajo busca desarrollar modelos de clasificación para poder apoyar la decisión de desembolsar un préstamo. La data que se empleará se encuentra disponible en este git 'datos_banco.csv'. La data cuenta con 14 atributos y 1719 instancias, donde el atributo a predecir 'target' que indica si una persona cayó en mora(1) o cumplió con los pagos(0).

## Análisis exploratorio

La data cuenta con 8 atributos cuantitativos y 6 cualitativos, incluyendo el primero. El primer atributo corresponde al indice de la instancia y no es utilizable para la clasificación.

### Data cualitativa

Se realizó un conteo de cada valor por cada atributo.
<p align="center"> <img src="Images/image_person_gender.png" width="350"/> <img src="Images/image_credit_history_marital_status.png" width="350"/> </p>
<p align="center"> <img src="Images/image_person_degree_type_desc.png" width="350"/> <img src="Images/image_var_max_sbs_qual_12m.png" width="350"/> </p>
