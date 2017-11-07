import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Main {

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val spark = SparkSession
      .builder()
      .appName("test")
      .master("local[*]")
      .getOrCreate()

    /**
      * Parse parameters
      */
    if (!args.contains("--input") || !args.contains("--output")) throw new Error(usageHelp)
    val input = args(args.indexOf("--input") + 1)
    val output = args(args.indexOf("--output") + 1)
    val model = args(args.indexOf("--model") + 1)

    val t1 = System.currentTimeMillis()
    args(0) match {
      case "train" => trainModel(spark, input, output)
      case "test" => testModel(spark, input, output, model, false)
    }
    println("Execution time: " + (System.currentTimeMillis() - t1).toString + "ms")
  }

  def usageHelp(): String = {
    return "\n==============================" +
      "\nHelp:" +
      "\nExecutable [COMMAND] [PARAMETERS]" +
      "\nCOMMAND: 'train' or 'test'" +
      "\nPARAMETERS" +
      "\n--input: /path/to/inputData" +
      "\n--model (if COMMAND is 'test'): /path/to/model" +
      "\n--output: /path/to/output.csv" +
      "\nInput parameter" +
      "\n - the dataset if the command is 'train'" +
      "\n - the dataset the model will be running against is 'test'" +
      "\nModel parameter" +
      "\n - The path to the model that will be tested (generated from training)" +
      "\nOutput parameter" +
      "\n - the trained model if the command is 'train'" +
      "\n - the predicted data csv if the command is 'test'" +
      "\n=============================="
  }

  def trainModel(spark: SparkSession, input: String, outputModel: String, modelType: String = "RANDOM_FOREST") = {
    val data: DataFrame = spark.read.format("libsvm").json(input)
      .transform(CleanProcess.cleanAndBalanceData(spark))

    println("=============================")
    println("=                           =")
    println("=       TRAINING MODEL      =")
    println("=                           =")
    println("=============================")

    Model.generateModel(modelType, outputModel)(data)
  }

  def testModel(spark: SparkSession, input: String, output: String, modelPath: String, debug: Boolean = true, modelType: String = "RANDOM_FOREST") = {
    val data: DataFrame = spark.read.format("libsvm").json(input)
      .transform(CleanProcess.cleanData(spark))
    println("=============================")
    println("=                           =")
    println("=       TESTING MODEL       =")
    println("=                           =")
    println("=============================")
    modelType match {
      case "RANDOM_FOREST" => Evaluators.evaluateRandomForest(RandomForestClassificationModel.load(modelPath), data, "label", "prediction", debug)
      case "LOGISTIC_REGRESSION" => Evaluators.evaluateLinearRegression(LogisticRegressionModel.load(modelPath), data, "label", "prediction").show(100)
      case "DECISION_TREE" => Evaluators.evaluateDecisionTree(DecisionTreeClassificationModel.load(modelPath), data, "label", "prediction").show(100)
      case default => println(s"Unexpected model: $default, please use one of RANDOM_FOREST, LOGISTIC_REGRESSION or DECISION_TREE")
    }
  }

  def writeCsv(output: String)(dataFrame: DataFrame) = {
    dataFrame.write
      .mode("overwrite")
      .format("com.databricks.spark.csv")
      .option("header", value = true)
      .option("delimiter", ",")
      .save(output)
    dataFrame
  }
}
