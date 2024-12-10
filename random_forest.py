from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Inicialização do Spark
spark = SparkSession.builder.appName('RandomForestKFold').getOrCreate()

# Carregar os dados
path = './data/treino_sinais_vitais_com_label.csv'
data = spark.read.csv(path, header=True, inferSchema=True)

# Definir as colunas de entrada e saída
input_cols = ["pSist", "pDiast", "qPa", "pulso", "respiracao"]
output_col = "rotulo"

# Transformar a coluna de rótulo em números
indexer = StringIndexer(inputCol="rotulo", outputCol="label")
data = indexer.fit(data).transform(data)

# Criar a coluna de características
assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
data = assembler.transform(data)

# Definir o classificador Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

# Definir o avaliador
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Definir os valores para o número de folds e número de árvores
folds_list = [3, 5, 7, 10]
trees_list = [50, 100, 150]

best_accuracy = 0
best_k = 0
best_trees = 0

# Testar combinações de folds e número de árvores
for k in folds_list:
    for trees in trees_list:
        # Construir o grid de parâmetros com o número de árvores
        paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [trees]).build()

        # Criar o CrossValidator para cada combinação de folds e numTrees
        crossval = CrossValidator(estimator=rf,
                                  evaluator=evaluator,
                                  estimatorParamMaps=paramGrid,
                                  numFolds=k)

        # Treinar o modelo com K-Fold Cross-Validation
        cv_model = crossval.fit(data)

        # Obter as previsões e calcular a precisão
        predictions = cv_model.bestModel.transform(data)
        accuracy = evaluator.evaluate(predictions)

        # Verificar se esta combinação obteve melhor precisão
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_trees = trees

# Exibir o melhor número de folds e árvores
print(f"Melhor número de folds: {best_k} e melhor número de árvores: {best_trees} com precisão: {best_accuracy:.4f}")
