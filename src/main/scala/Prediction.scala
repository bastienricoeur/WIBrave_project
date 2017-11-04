import org.apache.spark.ml.classification.{RandomForestClassifier}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame

object Prediction {

  def randomForest(features: String, label: String, prediction: String, metricName: String)(data: DataFrame): DataFrame = {
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val randomForest = new RandomForestClassifier()
      .setLabelCol(label)
      .setFeaturesCol(features)
      .setNumTrees(10)

    // Train model. This also runs the indexers.
    val model = randomForest.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    var result = predictions
      .select(prediction, label)
      .rdd
      .map { row => row.getAs[Double](prediction) -> row.getAs[Double](label) }
    val metrics = new MulticlassMetrics(result)
    println("confusion Matrix :")
    println(metrics.confusionMatrix.toString)
    print("truePositive rate :")
    println(metrics.weightedTruePositiveRate.toString())
    print("falsePositive rate :")
    println(metrics.weightedFalsePositiveRate.toString())
    predictions
  }
}