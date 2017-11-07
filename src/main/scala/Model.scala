import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


object Model {

  def decisionTree(label: String, features: String, prediction: String, metricName: String)(dataFrame: DataFrame): DecisionTreeClassificationModel = {

    val splitData = dataFrame.randomSplit(Array(0.7, 0.3))
    val dataToTrain = splitData(0)
    val dataToTest = splitData(1)

    val decisionTreeClassifier = new DecisionTreeClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setMaxDepth(30)
      .setMaxBins(15000)

    val classificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol(prediction)
      .setMetricName(metricName)

    val paramGrid = new ParamGridBuilder()
      .build()

    val crossValidator = new CrossValidator()
      .setNumFolds(10)
      .setEstimator(decisionTreeClassifier)
      .setEvaluator(classificationEvaluator)
      .setEstimatorParamMaps(paramGrid)

    val model = crossValidator
      .fit(dataToTrain)
      .bestModel
      .asInstanceOf[DecisionTreeClassificationModel]

    model
  }

  def randomForest(label: String, features: String, prediction: String)(data: DataFrame): RandomForestClassificationModel = {
    val randomForest = new RandomForestClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setNumTrees(1000)
      .setImpurity("gini")
      .setMaxBins(244)

    val model = randomForest.fit(data)
    model
  }

  def generateModel(modelType: String = "RANDOM_FOREST", modelPath: String)(dataFrame: DataFrame): Unit = {
    modelType match {
      case "RANDOM_FOREST" => Model.randomForest("label", "features", "prediction")(dataFrame).write.overwrite.save(modelPath)
      case "LOGISTIC_REGRESSION" => Model.logisticRegression("label", "features", "prediction")(dataFrame).write.overwrite.save(modelPath)
      case "DECISION_TREE" => Model.decisionTree("label", "features", "prediction", "areaUnderROC")(dataFrame).write.overwrite.save(modelPath)
      case default => println(s"Unexpected model: $default, please use one of RANDOM_FOREST, LOGISTIC_REGRESSION or DECISION_TREE")
    }
  }

  def logisticRegression(label: String, features: String, prediction: String, maxIter: Int = 10, regParam: Double = 0.3, elasticNetParam: Double = 0.8, debug: Boolean = true)(data: DataFrame): LogisticRegressionModel = {
    val logisticRegression = new LogisticRegression()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setPredictionCol(prediction)
      .setRawPredictionCol("rawPrediction")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setThreshold(0.5)
      .setFamily("auto")

    val model = logisticRegression.fit(data)
    model
  }

}