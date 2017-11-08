import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

object Evaluators {

  def evaluateDecisionTree(model: DecisionTreeClassificationModel, testData: DataFrame, label: String, prediction: String, metricName: String = "areaUnderROC"): DataFrame = {

    val predictions = model.transform(testData)

    val result = predictions
      .select(prediction, label)
      .rdd
      .map { row => row.getAs[Double](prediction) -> row.getAs[Double](label) }

    val metrics = new MulticlassMetrics(result)

    println("Confusion matrix")
    println(metrics.confusionMatrix)

    println("True positives")
    println(metrics.weightedTruePositiveRate)

    println("False positives")
    println(metrics.weightedFalsePositiveRate)

    println("Accuracy")
    println(metrics.accuracy)

    println("Precison")
    println(metrics.weightedPrecision)

    println("Recall")
    println(metrics.weightedRecall)

    predictions
  }

  def evaluateRandomForest(model: RandomForestClassificationModel, testData: DataFrame, label: String, prediction: String, debug: Boolean = true, metricName: String = "areaUnderROC"): DataFrame = {
    val predictions = model.transform(testData)

    if (debug) {
      val binaryClassificationEvaluator = new BinaryClassificationEvaluator().setLabelCol(label).setRawPredictionCol("rawPrediction")
      binaryClassificationEvaluator.setMetricName(metricName)
      val labelPrediction = predictions.select(label, prediction)
      val trueNeg = labelPrediction.filter(col(label) === 0.0 && col(prediction) === 0.0)
      val truePos = labelPrediction.filter(col(label) === 1.0 && col(prediction) === 1.0)
      val falseNeg = labelPrediction.filter(col(label) === 1.0 && col(prediction) === 0.0)
      val falsePos = labelPrediction.filter(col(label) === 0.0 && col(prediction) === 1.0)

      val nbTruePos = truePos.collect.size
      val nbFalsePos = falsePos.collect.size
      val nbFalseNeg = falseNeg.collect.size
      val nbTrueNeg = trueNeg.collect.size
      val nbEntries = labelPrediction.collect.size

      println(s"$nbTruePos CLICS WERE PREDICTED CORRECTLY")
      println(s"$nbFalseNeg CLICS WEREN'T PREDICTED AS CLICS")
      println(s"$nbFalsePos CLICS WERE PREDICTED BUT WEREN'T CLICS")
      println(s"$nbTrueNeg NOT CLICS WERE PREDICTED CORRECTLY")

      var result = predictions
        .select(prediction, label)
        .rdd
        .map { row => row.getAs[Double](prediction) -> row.getAs[Double](label) }
      val metrics = new BinaryClassificationMetrics(result)
      val multiMetrics = new MulticlassMetrics(result)

      metrics.precisionByThreshold.foreach { case (t, r) =>
        println(s"Threshold: $t, Precision: $r")
      }

      metrics.recallByThreshold.foreach { case (t, r) =>
        println(s"Threshold: $t, Recall: $r")
      }

      val accuracy = multiMetrics.accuracy
      println("Confusion matrix:")
      println(multiMetrics.confusionMatrix)
      println(s"Accuracy $accuracy")
    }

    predictions
  }

  def evaluateLinearRegression(model: LogisticRegressionModel, testData: DataFrame, label: String, prediction: String): DataFrame = {
    val binaryClassificationEvaluator = new BinaryClassificationEvaluator().setLabelCol(label).setRawPredictionCol("rawPrediction")
    binaryClassificationEvaluator.setMetricName("areaUnderROC")
    val regressionEvaluator = new RegressionEvaluator().setLabelCol(label).setPredictionCol(prediction)
    regressionEvaluator.setMetricName("rmse")

    val predictions = model.transform(testData)

    var result = predictions
      .select(prediction, label)
      .rdd
      .map { row => row.getAs[Double](prediction) -> row.getAs[Double](label) }


    // Instantiate metrics object
    val metrics = new MulticlassMetrics(result)

    // Confusion matrix
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    println("Precision = " + metrics.weightedPrecision)
    println("Recall = " + metrics.weightedRecall)

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    labels.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    labels.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }
    predictions
  }
}
