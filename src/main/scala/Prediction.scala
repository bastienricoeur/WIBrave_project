import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


object Prediction {

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

  def logisticRegression(label: String, features: String, prediction: String, maxIter: Int = 1, regParam: Double = 0.3, elasticNetParam: Double = 0.8, debug: Boolean = true)(data: DataFrame): LogisticRegressionModel = {
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

    val model = logisticRegression
      .fit(data)
    model
  }




}