import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}

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
    val dataFile = "/Users/benjaminafonso/Downloads/data-students.json"

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

    val data: DataFrame = spark.read.format("libsvm").json(dataFile).limit(20000)
      .transform(CleanProcess.label)
      .transform(CleanProcess.balanceDataset)
      .transform(CleanProcess.os)
      .transform(CleanProcess.bidFloor)
      .transform(CleanProcess.timestamp)
      .transform(CleanProcess.sizeBanner)
      .transform(CleanProcess.interests)
      .transform(CleanProcess.media)
      .transform(CleanProcess.weightDataset(spark))
      .transform(indexStrings(
        "os",
        "appOrSite",
        "bidFloor",
        "publisher",
        "timestamp",
        "sizeReset",
        "media",
        "label"
      ))
      .transform(vectorizeFeatures("features",
        "os_indexed",
        "appOrSite_indexed",
        "bidFloor_indexed",
        "publisher_indexed",
        "timestamp_indexed",
        "sizeReset_indexed",
        "media_indexed"
      ))
  }

  def indexStrings(columnNames: String*)(data: DataFrame): DataFrame = {
    val indexers = columnNames.map { columnName => {
      new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_indexed")
        .fit(data)
    }}
    indexers.foldLeft(data) { (df, indexer) => indexer.transform(df) }
  }

  def generateModel(dataFrame: DataFrame) = {
    Prediction.randomForest("label", "features", "prediction")
  }

  def vectorizeFeatures(vectorName: String, columnNames: String*)(dataFrame: DataFrame): DataFrame = {
    new VectorAssembler()
      .setInputCols(columnNames.map {_.toString }.toArray)
      .setOutputCol(vectorName)
      .transform(dataFrame)
  }

}
