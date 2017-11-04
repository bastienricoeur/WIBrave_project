import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Dataset, Row, SQLContext, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object SimpleScalaSpark {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val dataFile = "../data-students.json" // Should be some file on your system
    val spark = SparkSession.builder().appName("test").master("local[*]").getOrCreate()
    import spark.implicits._

    val data = spark.read.format("libsvm").json(dataFile)
      .withColumn("os", when(((!(lower($"os") === "ios") && !(lower($"os") === "android")) || $"os".isNull), "other").otherwise(lower($"os")))
      .withColumn("label", when($"label" === true, 1.0).otherwise(0.0))
      .withColumn("bidFloor", when($"bidFloor" < 2,"<2").otherwise($"bidFloor"))
      .withColumn("bidFloor", when($"bidFloor" >= 2 && $"bidFloor" <4,">=2 & <4").otherwise($"bidFloor"))
      .withColumn("bidFloor", when($"bidFloor" >= 4 && $"bidFloor" <10,">=4 & <10").otherwise($"bidFloor"))
      .withColumn("bidFloor", when($"bidFloor" >= 10,">=10").otherwise($"bidFloor"))
      .withColumn("bidFloor", when($"bidFloor".isNull,"null").otherwise($"bidFloor"))
      .withColumn("timestamp",$"timestamp".cast(TimestampType))

      .withColumn("timestamp",hour($"timestamp"))
      /*.withColumn("timestamp",when(hour($"timestamp")>=8 && hour($"timestamp")<12,"[8-12[").otherwise($"timestamp"))
      .withColumn("timestamp",when(hour($"timestamp")>=12 && hour($"timestamp")<18,"[12-18[").otherwise($"timestamp"))
      .withColumn("timestamp",when(hour($"timestamp")>=18,"[18[").otherwise($"timestamp"))
      .withColumn("timestamp",when(hour($"timestamp")<8,"]8]").otherwise($"timestamp"))*/
      .withColumn("sizeReset",when(!$"size".isNull,"notNull").otherwise("null"))
      .withColumn("sizeReset", when($"size"(0)<200 && $"size"(1)<200,"small").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(0)>200,"large").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(1)>200,"height").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(0)>200 && $"size"(1)>200,"medium").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(0)>300,"mediumlarge").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(1)>300,"mediumheight").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(0)>300 && $"size"(1)>300,"mediumBig").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(0)>400,"littlebiglarge").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(1)>400,"littlebigheight").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(0)>400 && $"size"(1)>400,"littleBig").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(0)>500,"biglarge").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(1)>500,"bigheight").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(0)>500 && $"size"(1)>500,"big").otherwise($"sizeReset"))
      .withColumn("sizeReset", when($"size"(0) === 320 && $"size"(1) === 480,"320*480").otherwise($"sizeReset"))
      .withColumn("C1", when($"interests".contains("IAB1-") || $"interests".contains("IAB1,") || $"interests" == "IAB1",1.0).otherwise(0.0))
      .withColumn("C2", when($"interests".contains("IAB2-") || $"interests".contains("IAB2,") || $"interests" == "IAB2",1.0).otherwise(0.0))
      .withColumn("C3", when($"interests".contains("IAB3"),1.0).otherwise(0.0))
      .withColumn("C4", when($"interests".contains("IAB4"),1.0).otherwise(0.0))
      .withColumn("C5", when($"interests".contains("IAB5"),1.0).otherwise(0.0))
      .withColumn("C6", when($"interests".contains("IAB6"),1.0).otherwise(0.0))
      .withColumn("C7", when($"interests".contains("IAB7"),1.0).otherwise(0.0))
      .withColumn("C8", when($"interests".contains("IAB8"),1.0).otherwise(0.0))
      .withColumn("C9", when($"interests".contains("IAB9"),1.0).otherwise(0.0))
      .withColumn("C10", when($"interests".contains("IAB10"),1.0).otherwise(0.0))
      .withColumn("C11", when($"interests".contains("IAB11"),1.0).otherwise(0.0))
      .withColumn("C12", when($"interests".contains("IAB12"),1.0).otherwise(0.0))
      .withColumn("C13", when($"interests".contains("IAB13"),1.0).otherwise(0.0))
      .withColumn("C14", when($"interests".contains("IAB14"),1.0).otherwise(0.0))
      .withColumn("C15", when($"interests".contains("IAB15"),1.0).otherwise(0.0))
      .withColumn("C16", when($"interests".contains("IAB16"),1.0).otherwise(0.0))
      .withColumn("C17", when($"interests".contains("IAB17"),1.0).otherwise(0.0))
      .withColumn("C18", when($"interests".contains("IAB18"),1.0).otherwise(0.0))
      .withColumn("C19", when($"interests".contains("IAB19"),1.0).otherwise(0.0))
      .withColumn("C20", when($"interests".contains("IAB20"),1.0).otherwise(0.0))
      .withColumn("C21", when($"interests".contains("IAB21"),1.0).otherwise(0.0))
      .withColumn("C22", when($"interests".contains("IAB22"),1.0).otherwise(0.0))
      .withColumn("C23", when($"interests".contains("IAB23"),1.0).otherwise(0.0))
      .withColumn("C24", when($"interests".contains("IAB24"),1.0).otherwise(0.0))
      .withColumn("C25", when($"interests".contains("IAB25"),1.0).otherwise(0.0))
      .withColumn("C26", when($"interests".contains("IAB26"),1.0).otherwise(0.0))

      .withColumn("media", when($"media".isNull,"null").otherwise($"media"))


    data.printSchema()
    //data.groupBy("C1").count.show
    val osIndexer = new StringIndexer().setInputCol("os").setOutputCol("indexedOs")
    val appOrSiteIndexer = new StringIndexer().setInputCol("appOrSite").setOutputCol("indexedAppOrSite")
    val bidFloorIndexer = new StringIndexer().setInputCol("bidFloor").setOutputCol("indexedBidFloor")
    val timeIndexer = new StringIndexer().setInputCol("timestamp").setOutputCol("indexedTimestamp")
    val sizeIndexer = new StringIndexer().setInputCol("sizeReset").setOutputCol("indexedSize")
    val typeIndexer = new StringIndexer().setInputCol("type").setOutputCol("indexedType")
    val mediaIndexer = new StringIndexer().setInputCol("media").setOutputCol("indexedMedia")




    //val cityIndexer = new StringIndexer().setInputCol("city").setOutputCol("indexedCity")
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    val assembler = new VectorAssembler()
      .setInputCols(Array("indexedOs","indexedAppOrSite","indexedBidFloor","indexedTimestamp","indexedSize","indexedMedia","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26"))
      .setOutputCol("indexedFeatures")

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Train a DecisionTree model.
    val dt = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)


    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(osIndexer,appOrSiteIndexer,bidFloorIndexer,timeIndexer,sizeIndexer,mediaIndexer,labelIndexer, assembler, dt, labelConverter))
    val result = pipeline.fit(data).transform(data)

    /*println(":: START Chi2 selector")
    val selector = new ChiSqSelector()
      .setNumTopFeatures(3)
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setOutputCol("selectedFeatures")
    println(":: END Chi2 selector")

    val chi = selector.fit(result).transform(result)

    println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
    chi.show()*/
    // Train model. This also runs the indexers.
   val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)



    // Select example rows to display.
    var result2 = predictions.select("prediction", "label").rdd.map { row => row.getAs[Double]("prediction") -> row.getAs[Double]("label") }
    val metrics = new MulticlassMetrics(result2)
    println("confusion Matrix :")
    println(metrics.confusionMatrix.toString)
    print("truePositive rate :")
    println(metrics.weightedTruePositiveRate.toString())
    print("falsePositive rate :")
    println(metrics.weightedFalsePositiveRate.toString())

    val treeModel = model.stages(8).asInstanceOf[RandomForestClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)
/*
    // Select (prediction, true label) and compute test error.
    val evaluator1 = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
    val evaluator2 = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderPR")
    val auroc = evaluator1.evaluate(predictions)
    val aupr = evaluator2.evaluate(predictions)

    println("AUROC: "+auroc+" || AUPR: "+aupr)

    val treeModel = model.stages(7).asInstanceOf[RandomForestClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)*/
  }

}

case class Flow(appOrSite: String, bidfloor: Double, city: String, exchange: String, impid: String, interests: String, label: Boolean, media: String, publisher: String, os: String, network: String, size: Array[Long], timestamp: Long, typ: String, user: String)