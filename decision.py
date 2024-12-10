from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import col, when, count, lit, expr
from numbers import Number

from sklearn.model_selection import KFold
import random
import itertools
import numpy as np
from datetime import datetime

# Inicializa a sessão Spark
spark = SparkSession.builder.appName("DecisionTreeSpark").getOrCreate()

class DecisionTree:
    """
    Implementacao uma árvore de decisão utilizando PySpark.
    """

    def __init__(self, min_samples_split=2, max_depth=None, seed=65):
        # Define parâmetros principais para o treinamento da árvore.
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.seed = seed
        self.children = {}
        self.split_feature = None
        self.threshold = None
        self.decision = None

    def configure(self, params):
        self.min_samples_split = params.get("min_samples_split", 2)
        self.max_depth = params.get("max_depth", None)
        self.seed = params.get("seed", 62)

    def fit(self, df: DataFrame, target_col: str, depth=0):
        """
        Treina a árvore de decisão de forma recursiva, criando nós e folhas.
        :param df: DataFrame PySpark com os dados de treino.
        :param target_col: Nome da coluna-alvo.
        :param depth: Profundidade atual da árvore.
        """
        unique_classes = df.select(target_col).distinct().count()

        # Caso base: apenas uma classe restante no nó.
        if unique_classes == 1:
            self.decision = df.select(target_col).first()[0]
            return

        # Caso base: número mínimo de amostras ou profundidade máxima.
        if df.count() < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            self.decision = self._majority_class(df, target_col)
            return

        # Realiza a divisão do nó em atributos.
        best_feature, best_threshold, splits = self._find_best_split(df, target_col)
        if not best_feature:
            self.decision = self._majority_class(df, target_col)
            return

        # Atribui os valores do melhor split encontrado.
        self.split_feature = best_feature
        self.threshold = best_threshold

        # Cria subnós de forma recursiva.
        self.children = {}
        for value, subset in splits.items():
            if subset.count() > 0:
                child = DecisionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    seed=self.seed
                )
                child.fit(subset, target_col, depth + 1)
                self.children[value] = child

    def predict(self, sample: dict):
        """
        Realiza a predição para uma única amostra com base na árvore treinada.
        :param sample: Dicionário contendo os atributos da amostra.
        :return: Classe prevista.
        """
        if self.decision is not None:
            return self.decision

        value = sample.get(self.split_feature)
        if isinstance(value, Number) and self.threshold is not None:
            value = "greater" if value > self.threshold else "less"

        child = self.children.get(value)
        if child:
            return child.predict(sample)
        return self.decision

    def _majority_class(self, df: DataFrame, target_col: str):
        """
        Calcula a classe majoritária no conjunto de dados.
        :param df: DataFrame PySpark com os dados.
        :param target_col: Nome da coluna-alvo.
        :return: Classe majoritária.
        """
        counts = df.groupBy(target_col).count().orderBy(col("count").desc()).collect()
        return counts[0][0]

    def _find_best_split(self, df: DataFrame, target_col: str):
        """
        Identifica o melhor atributo e threshold para realizar o split.
        :param df: DataFrame PySpark com os dados.
        :param target_col: Nome da coluna-alvo.
        :return: Melhor atributo, threshold e os subsets divididos.
        """
        info_gain_max = -float("inf")
        best_feature = None
        best_threshold = None
        best_splits = None

        for column in df.columns:
            if column == target_col:
                continue

            if df.schema[column].dataType == StringType():
                # Split para atributos categóricos
                splits = {val: df.filter(col(column) == val) for val in df.select(column).distinct().collect()}
                info_gain = self._information_gain(df, target_col, splits)

            elif df.schema[column].dataType == DoubleType():
                # Split para atributos contínuos
                thresholds = df.select(column).distinct().orderBy(column).rdd.map(lambda x: x[0]).collect()
                splits = {}
                for threshold in thresholds:
                    splits["less"] = df.filter(col(column) <= threshold)
                    splits["greater"] = df.filter(col(column) > threshold)
                    info_gain = self._information_gain(df, target_col, splits)

            if info_gain > info_gain_max:
                info_gain_max = info_gain
                best_feature = column
                best_threshold = threshold if 'threshold' in locals() else None
                best_splits = splits

        return best_feature, best_threshold, best_splits

    def _information_gain(self, df: DataFrame, target_col: str, splits: dict):
        """
        Calcula o ganho de informação para um dado split.
        :param df: DataFrame PySpark com os dados.
        :param target_col: Nome da coluna-alvo.
        :param splits: Dicionário com os subsets divididos.
        :return: Ganho de informação calculado.
        """
        total_entropy = self._entropy(df, target_col)
        split_entropy = sum(
            (subset.count() / df.count()) * self._entropy(subset, target_col)
            for subset in splits.values()
        )
        return total_entropy - split_entropy

    def _entropy(self, df: DataFrame, target_col: str):
        """
        Calcula a entropia do conjunto de dados.
        :param df: DataFrame PySpark com os dados.
        :param target_col: Nome da coluna-alvo.
        :return: Entropia calculada.
        """
        counts = df.groupBy(target_col).count().rdd.map(lambda x: x[1]).collect()
        probs = np.array(counts) / sum(counts)
        return -np.sum(probs * np.log2(probs + 1e-6))



def evaluate_model(data, labels, model, folds, parameters):
    kfold = KFold(n_splits=folds)
    param_combinations = [dict(zip(parameters.keys(), values)) for values in itertools.product(*parameters.values())]
    scores = []

    for param in param_combinations:
        model.configure(param)
        for train_idx, test_idx in kfold.split(data):
            train_data, train_labels = data.iloc[train_idx], labels.iloc[train_idx]
            test_data, test_labels = data.iloc[test_idx], labels.iloc[test_idx]
            model.train(train_data, train_labels)
            predictions = test_data.apply(model.classify, axis=1)
            accuracy = (predictions == test_labels).mean()
            scores.append(accuracy)
    return scores
