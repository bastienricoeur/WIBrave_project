import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.feature._

object SimpleScalaSpark {

  def usageHelp(): String = {
    return "\n==============================" +
      "\nHelp:" +
      "\n--inputFile : /path/to/inputData" +
      "\n=============================="
  }

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    /**
      * Parse parameters
      */
    //val dataFile = args(args.indexOf("--inputFile"))
    val dataFile = "/Users/bastienricoeur/Desktop/wi_lib/data-students.json"

    if (!args.contains("--inputFile")) {
      //throw new Error(usageHelp)
      //return
    }

    // Initialize SparkSession
    val spark = SparkSession
      .builder()
      .appName("test")
      .master("local[*]")
      .getOrCreate()

    val data: DataFrame = spark.read.format("libsvm").json(dataFile).limit(1000)
      .transform(CleanProcess.os)
      .transform(CleanProcess.label)
      .transform(CleanProcess.bidFloor)
      .transform(CleanProcess.timestamp)
      .transform(CleanProcess.sizeBanner)
      .transform(CleanProcess.interests)
      .transform(CleanProcess.media)
      .transform(indexStrings(
        "os",
        "appOrSite",
        "bidFloor",
        "timestamp",
        "sizeReset",
        "media",
        "label"
      ))
      .transform(vectorizeFeatures("features",
        "os_indexed",
        "appOrSite_indexed",
        "bidFloor_indexed",
        "timestamp_indexed",
        "sizeReset_indexed",
        "media_indexed"
      ))
      .transform(Prediction.randomForest("features", "label_indexed", "prediction", ""))
  }

  def indexStrings(columnNames: String*)(dataFrame: DataFrame): DataFrame = {
    val indexers = columnNames.map { columnName => {
      new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_indexed")
        .fit(dataFrame)
    }}
    indexers.foldLeft(dataFrame) { (df, indexer) => indexer.transform(df) }
  }

  def vectorizeFeatures(vectorName: String, columnNames: String*)(dataFrame: DataFrame): DataFrame = {
    new VectorAssembler()
      .setInputCols(columnNames.map {_.toString }.toArray)
      .setOutputCol(vectorName)
      .transform(dataFrame)
  }

}

case class Flow(appOrSite: String, bidfloor: Double, city: String, exchange: String, impid: String, interests: String, label: Boolean, media: String, publisher: String, os: String, network: String, size: Array[Long], timestamp: Long, typ: String, user: String)