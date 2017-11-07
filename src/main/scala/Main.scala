import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.sql.{DataFrame, SparkSession}

object Main {

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val spark = SparkSession
      .builder()
      .appName("test")
      .master("local[*]")
      .getOrCreate()

    /**
      * Parse parameters
      */
    if (!args.contains("--input") || args.contains("--output")) throw new Error(usageHelp)
    val input = args(args.indexOf("--input") + 1)
    val output = args(args.indexOf("--output") + 1)
    val model = args(args.indexOf("--model") + 1)

    val data: DataFrame = spark.read.format("libsvm").json(input).limit(1000)
      .transform(CleanProcess.cleanData(spark))

    args(0) match {
      case "train" => trainModel(data, output)
      case "test" => testModel(data, output, model, spark)
    }
  }

  def usageHelp(): String = {
    return "\n==============================" +
      "\nHelp:" +
      "\nExecutable [COMMAND] [PARAMETERS]" +
      "\nCOMMAND: 'train' or 'test'" +
      "\nPARAMETERS" +
      "\n--input: /path/to/inputData" +
      "\n--model: /path/to/model" +
      "\n--output: /path/to/output.csv" +
      "\nInput parameter" +
      "\n - the dataset if the command is 'train'" +
      "\n - the dataset the model will be running against is 'test'" +
      "\nOutput parameter" +
      "\n - the trained model if the command is 'train'" +
      "\n - the predicted data csv if the command is 'test'" +
      "\n=============================="
  }

  def trainModel(dataFrame: DataFrame, outputModel: String, modelType: String = "RANDOM_FOREST") = {
    Model.generateModel(modelType, outputModel)(dataFrame)
  }

  def testModel(dataFrame: DataFrame, output: String, modelPath: String, spark: SparkSession, modelType: String = "RANDOM_FOREST") = {
    modelType match {
      case "RANDOM_FOREST" => RandomForestClassificationModel.load(modelPath)
      case "LOGISTIC_REGRESSION" => LogisticRegressionModel.load(spark.sparkContext, modelPath)
      case "DECISION_TREE" => DecisionTreeClassificationModel.load(modelPath)
      case default => println(s"Unexpected model: $default, please use one of RANDOM_FOREST, LOGISTIC_REGRESSION or DECISION_TREE")
    }
  }
}
