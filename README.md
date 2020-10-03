# Informe Trabajo Parcial Machine Learning

## Integrantes:
- Alegre Flores, Renzo Paolo
- Baldeón Albornoz, Braulio Sebastián
- Nuñez Robinson, Daniel

## Introducción

El presente trabajo busca desarrollar modelos de clasificación para poder apoyar la decisión de desembolsar un préstamo. La data que se empleará se encuentra disponible en este git 'datos_banco.csv'. La data cuenta con 14 atributos y 1719 instancias, donde el atributo a predecir 'target' que indica si una persona cayó en mora(1) o cumplió con los pagos(0).

## Método

### Análisis Exploratorio

El dataset con el que se trabajará es estructurado y cuenta con 8 atributos cuantitativos y 6 cualitativos, incluyendo el primero que corresponde al indice de la instancia. Por lo tando, no es necesario considerarla durante la elaboración de los modelos.

#### Data cualitativa

Se realizó un conteo de la cantidad de instancias según valor por cada atributo cualitativo.
<p align="center"> <img src="Images/Análisis Exploratorio/image_person_gender.png" width="350"/> <img src="Images/Análisis Exploratorio/image_credit_history_marital_status.png" width="350"/> </p>
<pre align="center"> Figura 1                                        Figura 2</pre>
<p align="center"> <img src="Images/Análisis Exploratorio/image_person_degree_type_desc.png" width="350"/> <img src="Images/Análisis Exploratorio/image_var_max_sbs_qual_12m.png" width="350"/> </p>
<pre align="center"> Figura 3                                        Figura 4</pre>

Como se en la Figura 4, el valor 'Carrera técnica' del atributo 'person_degree_type_desc' se ha separado en dos por los caracteres especiales. Por lo tanto, un paso del pre-procesamiento será juntar estas instancias.

### Data cuantitativa

<p align="center"> <img src="Images/Análisis Exploratorio/reporte data.PNG" width="400"/> </p>
<pre align="center"> Figura 5 </pre>

Como se observa en la Figura 5, el unico atributo con missing values es 'var_employment_time_in_months' y que algunos atributos cuantitativos son tipo 'object', esto es debido q que los valores mayor a 1000 son representados con "," y almacenados como 'string'. Por lo tanto, un paso del pre-procesamiento será el tratamiento de missing values y la conversión de los atributos tipo 'object'

<p align="center"> <img src="Images/Análisis Exploratorio/image_default_request_amount.png" width="450"/> </p>
<pre align="center"> Figura 6 </pre>

Como se observa en la Figura 6, el atributo 'image_default_request_amount' solo tiene instancias con valor 0. Por ende, debe ser eliminado en el pre-procesamiento porque no tiene data utilizable para la tarea de clasificación.

<p align="center"> <img src="Images/Análisis Exploratorio/grafico de correlacion.png" width="450"/> </p>
<pre align="center"> Figura 7 </pre>

La Figura 7 es un gráfico de correlación entre los atributos cualitativos. Como se observa hay una alta correlación entre los atributos 'approved_amount', 'appreove_term', 'approved_interest_amount'. Lo cual es lógico porque a mayor monto de prestamo, mayor el tiempo para pagarlo y mayor el interes. Por otro lado, hay baja correlación entre los otras atributos.

También se uso la libreria seaborn para graficar un pairplot, el cual muestra la relación para (n, 2) combinación de variables en un DataFrame como una matriz de gráficos y los gráficos diagonales son los gráficos univariados

<p align="center"> <img src="Images/Análisis Exploratorio/pair plot.png" width="600"/> </p>
<pre align="center"> Figura 8 </pre>

Como se observa en la Figura 8, los gráficos univariados evidencia la existencia de outliers. Por lo tanto, un paso de del pre-procesamiento debe ser normalizar estos atributos.

### Tratamiento de datos

Lo primero que se hizo fue eliminar los atributos con data no utilizable para la clasificación: 'idPrestamo' y 'default_request_amount'. El primero por ser un indice y el segundo porque el dataset solo contiene instancias de valor '0'.

En segundo lugar se recorrió el atributo 'person_degree_type_desc' par agrupar las instancias de valor 'Carrera Tecnica' que habian sido separadas en dos por lo caracteres especiales.

Después, se procedio a utilizar OneHotEncoding para transformar los atributos cualitativos en cuantitativos. Este metodo convierte la data cualitativa en un arreglo binario, como se puede observar en la Figura 9, para que el modelo pueda realizar una mejor predicción.

<p align="center"> <img src="Images/Pre-Procesamiento/OneHotEncoding.PNG" width="450"/> </p>
<pre align="center"> Figura 9 </pre>

Luego, se transformó la data tipo 'object' en data numérica. Para lo cual, se tuvo que convertir los valores mayores a 1000 que estaban almacenados como 'string' en 'float' para, posteriormente, trasnformarlos con el metodo 'pd.to_numeric'.

En quinto lugar, se empleó SimpleImputer, de la libreria 'sklearn', con estrategia de media para el tratamiento de missing values. El unico atributo en el dataset que contenía NaN era 'var_employment_time_in_months'.

Finalmente, debido a que en el análisis se identificó la precencia de outliers se realizó la normalización de la data originalmente cuantitativa. Se utilizo MinMaxScaler de la librería 'sklearn' en los siguientes atributos 'monthly_debt_capacity', 'var_employment_time_in_months', 'approved_amount', 'approved_interest_amount' y 'var_net_incom'.

### Validación de modelos 

#### Decision Tree Classifier

Modelo de clasificación supervisada donde los datos se dividen continuamente de acuerdo con un parámetro, los arboles de decisión constan de:
- Nodos, evaluan el valor de un determinado atributo.
- Ramas, corresponde al resultado de la evaluación y se conectan al siguiente nodo.
- Nodos hoja, nodos finales que corresponden a un valor del atributo objetivo (clase a predecir).

Los pasos que se seguimos para la construcción y validación de este modelo fueron los siguientes:

Primero, separamos nuestra data con una proporción 90/10 para probar el rendimiento de un DecisionTreeClassifier básico, implementado con la libreria sklearn.
<p align="center"> <img src="Images/Decision Tree/metricas modelo base.PNG" width="400"/> <img src="Images/Decision Tree/CV modelo base.PNG" width="300"/> </p>
<pre align="center"> Figura 9                                        Figura 10</pre>

Como se observa en la Figura 9, un árbol de decisión base obtuvo un 62% de accuracy con nuestra data. Así mismo, es notorio que el modelo no clasifica correctamente las instacias clase '1', posiblemente debido a que hay poca data de esta clase. También, se empleó 5-fold Cross-validation para evaluar el comportamiento del modelo con diferentes particiones del dataset. La Figura 10 muestra el accuracy obtenido en cada fold y la media de estos.

En segundo lugar, se graficaron curvas de validación y de aprendizaje para el proceso de validación del modelo. 

<p align="center"> <img src="Images/Decision Tree/Validation Curve max_depth.png" width="350"/> <img src="Images/Decision Tree/Validation Curve min_samples_leaf.png" width="350"/> </p>
<pre align="center"> Figura 11                                        Figura 12</pre>

La Figura 11 muestra la curva de validación para el hiperparámetro 'max_depth'. Como se puede observar el mejor accuracy conseguido con una produndidad máxima 3 y después de esta la curva de entrenamiento y validación de sepraran incrementalmente, lo que indica que existe una alta varianza a mayo profundidad. 

Por otro lado, la Figura 12 muestra la curva de validación para el hiperparámetro 'min_samples_leaf'. A diferencia de la curva anterior, si el parametro es pequeño surje una alta varianza y disminuye a mayor valor de parámetro. Específicamente, los valores 100, 110 y 120 son los que tiene mayor accuracy y los más cercanos a la curava de entrenamiento.

<p align="center"> <img src="Images/Decision Tree/Learning Curve.png" width="400"/> </p>
<pre align="center"> Figura 13 </pre>

La Figura 13 muestra la curva de aprendizaje del modelo, se observa que la cura de entrenamiento es muy cercana a 100% de accuracy lo cual es señal de overfitting. Por otro lado, la curva del cross-validation es muy variable a lo largo de las iteraciones. Además, entre las curvas hay mucha diferencia lo que indica un escenario de alta varianza. Es recomendable consegir más data o reducir la complejidad del modelo para mejorar el performance de este.

#### Neural Network (1er Dataset)

Modelos computacionales inspirados en el cerebro humano
- Neuronas: unidades de procesamiento
- Sinapsis: conexiones que intercomunican las neuronas

Los pasos que se seguimos para la construcción y validación de este modelo fueron los siguientes:
Para realizar el modelo se usaron las bibliotecas: pandas, numpy, matplotlib y sklearn.
Primero, el dataset es repartido en 4 porciones de prueba y los llamamos: 
"X_train, X_test, Y_train, Y_test"
Seguido, aplicamos KFold (10 splits o particiones).

La red neuronal fue ejecutada varias veces iterando sus atributos tales como número de capas, número de neuronas,el valor de alpha, etx.
Sin embargo, por más que estos hayan variado, el resultado fue muy similar. siendo el mejor promedio: 0.6489181194770927.

En la figura 14, se aprecioa la siguiente configuración de red neuronal:
red_neuronal = MLPClassifier(hidden_layer_sizes=(8,2), max_iter=100, learning_rate_init=0.01, activation='logistic')


<p align="center"> <img src="Images/RedesNeuronales/img1.PNG" width="450"/> </p>
<pre align="center"> Figura 14 </pre>

En la figura 15, se aprecioa la siguiente configuración de red neuronal (con alpha = 0.2) :
red_neuronal2 = MLPClassifier(hidden_layer_sizes=(8,2), max_iter=100, learning_rate_init=0.01, activation='logistic', alpha= 0.2)

<p align="center"> <img src="Images/RedesNeuronales/img2.PNG" width="450"/> </p>
<pre align="center"> Figura 15 </pre>

En la figura 16, se observa comportamiento que tuvo la neruona con la alteración de alpha = 0,2.

<p align="center"> <img src="Images/RedesNeuronales/img3.PNG" width="450"/> </p>
<pre align="center"> Figura 16 </pre>

En la figura 17 se puede apreciar los resultados de cada uno de los 10 folds de la red neuronal 2 (alpha = 0.2)
Este es el mejor resultado de todos las muestras que se probaron.

<p align="center"> <img src="Images/RedesNeuronales/img_fold.PNG" width="450"/> </p>
<pre align="center"> Figura 17 </pre>

Es necesario precisar que el valor promedio mayor fue 0.6489181194770927 el cual no es un valor muy favorable para aplicar, asimismo,
es necesario precisar que con un entrenemiento de mayor tiempo, el resultado puede incrementar y se puede obtener el mejor modelo iterando
alpha, número de capas y neuronas.


#### Modelo 3

## Resultados

Se empleó el metodo GridSearchCV para buscar la combinación de valores de hiperparámetros, previamente especifivados, que obtenga mejor acurracy. Los valores fueron acotados mediante la observación de las curvas de validación. Despues de obtener el mejor modelo, se realizó la predicción con la misma partición de data que se uso para el modelo base.

### Decision Tree Classifier

Los hiperparámetros sobre los que se iteró para este modelo fueron 'criterion', 'max_depth' y 'min_samples_leaf'.

<p align="center"> <img src="Images/Decision Tree/metricas best modelo.PNG" width="400"/> <img src="Images/Decision Tree/CV best modelo.PNG" width="300"/> </p>
<pre align="center"> Figura 18                                        Figura 19</pre>

Como se observa en la Figura 18, el modelo obtuvo un 72% de accuracy superando al modelo base. También, obtuvo un mayor valor de precision para la calse '1' lo que indica que el modelo pudo clasificar correctamenta una mayor cantidad de instacias de esta clase. Por otro lado, los resultados del 5-fold Cross-validation (Figura 18) son superiores a los del modelo base y hay una menor variación entre el resultado de cada fold.

### Modelo 2
### Modelo 3

## Conclusiones
