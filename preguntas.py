"""
Detección de hongos venenosos usando Regresión Logistica
-----------------------------------------------------------------------------------------

Construya un modelo de regresión logística que permita identificar si un hongo es 
venenoso o no. Para ello, utilice la muestra de datos suministrada. 

La base de datos contiene 8124 instancias de hongos provenientes de 23 especies de la 
familia Agaricus y Lepiota, los cuales han sido clasificados como comestibles, venenosos
o de comestibilidad indeterminada. Por el tipo de problema en cuestión, los hongos de 
comestibilidad desconocida deben ser asignados a la clase de hongos venenosos, ya que no
se puede correr el riesgo de dar un hongo potencialmente venenoso a una persona para su 
consumo.

Véase https://www.kaggle.com/uciml/mushroom-classification

Evalue el modelo usando la matriz de confusión.

La información contenida en la muestra es la siguiente:

     1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
                                  knobbed=k,sunken=s
     2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
     3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
                                  pink=p,purple=u,red=e,white=w,yellow=y
     4. bruises?:                 bruises=t,no=f
     5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
                                  musty=m,none=n,pungent=p,spicy=s
     6. gill-attachment:          attached=a,descending=d,free=f,notched=n
     7. gill-spacing:             close=c,crowded=w,distant=d
     8. gill-size:                broad=b,narrow=n
     9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
                                  green=r,orange=o,pink=p,purple=u,red=e,
                                  white=w,yellow=y
    10. stalk-shape:              enlarging=e,tapering=t
    11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
                                  rhizomorphs=z,rooted=r,missing=?
    12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    16. veil-type:                partial=p,universal=u
    17. veil-color:               brown=n,orange=o,white=w,yellow=y
    18. ring-number:              none=n,one=o,two=t
    19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
                                  none=n,pendant=p,sheathing=s,zone=z
    20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
                                  orange=o,purple=u,white=w,yellow=y
    21. population:               abundant=a,clustered=c,numerous=n,
                                  scattered=s,several=v,solitary=y
    22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
                                  urban=u,waste=w,woods=d


"""

import pandas as pd


def pregunta_01():
    """
    En esta función se realiza la carga de datos.
    """
    # Lea el archivo `mushrooms.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("mushrooms.csv", sep=",")

    # Remueva la columna `veil-type` del DataFrame `df`.
    # Esta columna tiene un valor constante y no sirve para la detección de hongos.
    df.pop("veil_type")

    # Asigne la columna `type` a la variable `y`.
    y = df['type']

    # Asigne una copia del dataframe `df` a la variable `X`.
    X = df.copy()

    # Remueva la columna `type` del DataFrame `X`.
    X.pop('type')

    # Retorne `X` y `y`
    return X, y


def pregunta_02():
    """
    Preparación del dataset.
    """

    # Importe train_test_split
    from sklearn.model_selection import train_test_split

    # Cargue los datos de ejemplo y asigne los resultados a `X` y `y`.
    X, y = pregunta_01()

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 123. Use 50 patrones para la muestra de prueba.
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size=(50/X.shape[0]),
        random_state=123,
    )

    # Retorne `X_train`, `X_test`, `y_train` y `y_test`
    return X_train, X_test, y_train, y_test


def pregunta_03():
    """
    Especificación y entrenamiento del modelo. En sklearn, el modelo de regresión
    logística (a diferencia del modelo implementado normalmente en estadística) tiene
    un hiperparámetro de regularición llamado `Cs`. Consulte la documentación.

    Para encontrar el valor óptimo de Cs se puede usar LogisticRegressionCV.

    Ya que las variables explicativas son literales, resulta más conveniente usar un
    pipeline.
    """

    # Importe LogisticRegressionCV
    # Importe OneHotEncoder
    # Importe Pipeline
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline

    # Cargue las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Cree un pipeline que contenga un estimador OneHotEncoder y un estimador
    # LogisticRegression con una regularización Cs=10
    pipeline = Pipeline(
        steps=[
            ("OneHot", OneHotEncoder()),
            ("LogReg", LogisticRegressionCV(Cs=10)),
        ],
    )

    # Entrene el pipeline con los datos de entrenamiento.
    pipeline.fit(X_train, y_train)

    # Retorne el pipeline entrenado
    return pipeline


def pregunta_04():
    """
    Evalue el modelo obtenido.
    """

    # Importe confusion_matrix
    from sklearn.metrics import confusion_matrix

    # Obtenga el pipeline de la pregunta 3.
    pipeline = pregunta_03()

    # Cargue las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Evalúe el pipeline con los datos de entrenamiento usando la matriz de confusion.
    cfm_train = confusion_matrix(
        y_true=y_train,
        y_pred=pipeline.predict(X_train),
    )

    cfm_test = confusion_matrix(
        y_true=y_test,
        y_pred=pipeline.predict(X_test),
    )

    # Retorne la matriz de confusion de entrenamiento y prueba
    return cfm_train, cfm_test
