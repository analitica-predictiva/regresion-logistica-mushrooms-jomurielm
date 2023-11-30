"""
Calificación del laboratorio
-----------------------------------------------------------------------------------------
"""

import sys

import preguntas


def test_01():
    # ---< Input/Output test case >----------------------------------------------------
    # Pregunta 01
    # pip3 install scikit-learn==1.0.2 pandas==1.1.5 numpy
    # python3 tests.py 01

    X, y = preguntas.pregunta_01()
    assert X.shape == (8124, 21)
    assert y.shape == (8124,)
    assert "veil_type" not in X.columns
    assert "type" not in X.columns


def test_02():
    # ---< Input/Output test case >----------------------------------------------------
    # Pregunta 02
    # pip3 install scikit-learn==1.0.2 pandas==1.1.5 numpy
    # python3 tests.py 02
    # (139, 10)
    # -0.7869
    # 69.6029
    # <class 'pandas.core.series.Series'>
    # 0.629

    X_train, X_test, y_train, y_test = preguntas.pregunta_02()
    assert y_train.value_counts().to_dict() == {"e": 4179, "p": 3895}
    assert y_test.value_counts().to_dict() == {"e": 29, "p": 21}
    assert X_train.iloc[:, 0].value_counts().to_dict() == {
        "x": 3628,
        "f": 3139,
        "k": 823,
        "b": 448,
        "s": 32,
        "c": 4,
    }
    X_train.iloc[:, 1].value_counts().to_dict() == {
        "y": 3219,
        "s": 2542,
        "f": 2309,
        "g": 4,
    }
    assert X_test.iloc[:, 1].value_counts().to_dict() == {"y": 25, "s": 14, "f": 11}


def test_03():
    # ---< Run command >-----------------------------------------------------------------
    # Pregunta 03
    # pip3 install scikit-learn==1.0.2 pandas==1.1.5 numpy
    # python3 tests.py 03

    X_train, X_test, y_train, y_test = preguntas.pregunta_02()
    pipeline = preguntas.pregunta_03()
    assert pipeline.score(X_train, y_train).round(6) == 1.0
    assert pipeline.score(X_test, y_test).round(6) == 1.0


def test_04():
    # ---< Run command >--------------------------------------------------------------------
    # Pregunta 04
    # pip3 install scikit-learn==1.0.2 pandas==1.1.5 numpy
    # python3 tests.py 04

    cfm_train, cfm_test = preguntas.pregunta_04()
    assert cfm_train.tolist() == [[4179, 0], [0, 3895]]
    assert cfm_test.tolist() == [[29, 0], [0, 21]]


test = {
    "01": test_01,
    "02": test_02,
    "03": test_03,
    "04": test_04,
}[sys.argv[1]]

test()
