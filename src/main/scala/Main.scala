import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, monotonically_increasing_id, when}

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
    if(args.contains("--help")){
      println(usageHelp())
    }else{
      if (!args.contains("--input") || !args.contains("--output")) throw new Error(usageHelp)
      val input = args(args.indexOf("--input") + 1)
      val output = args(args.indexOf("--output") + 1)
      val model = args(args.indexOf("--model") + 1)

      val t1 = System.currentTimeMillis()
      args(0) match {
        case "train" => trainModel(spark, input, output)
        case "test" => testModel(spark, input, output, model, true)
        case "predict" => testModel(spark, input, output, model, false)
      }
      println("Execution time: " + (System.currentTimeMillis() - t1).toString + "ms")
    }
  }

  def usageHelp(): String = {
    return "\n==============================" +
      "\nHelp:" +
      "\nExecutable [COMMAND] [PARAMETERS]" +
      "\nCOMMAND: 'train' or 'test' or 'predict'" +
      "\nPARAMETERS" +
      "\n--input: /path/to/inputData" +
      "\n--model (if COMMAND is 'test or predict'): /path/to/model" +
      "\n--output: /path/to/" +
      "\nInput parameter" +
      "\n - the dataset if the command is 'train'" +
      "\n - the dataset which will be tested or predicted if the command is 'test' or 'predict'" +
      "\nModel parameter" +
      "\n - The path to the model that will be tested (generated from training)" +
      "\nOutput parameter" +
      "\n - the trained model folder if the command is 'train'" +
      "\n - the folde with the predicted data csv inside if the command is 'test or predict'" +
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
    if(modelPath == "predict" || modelPath == "test"){
      throw new Error("Missing your model path")
    }
    val data: DataFrame = spark.read.format("libsvm").json(input).transform(CleanProcess.cleanData(spark,debug))

    if(debug){
      println("=============================")
      println("=                           =")
      println("=       TESTING MODEL       =")
      println("=                           =")
      println("=============================")
    }else{
      println("=============================")
      println("=                           =")
      println("=     PREDICTING MODEL      =")
      println("=                           =")
      println("=============================")
    }

    modelType match {
      case "RANDOM_FOREST" => Evaluators.evaluateRandomForest(RandomForestClassificationModel.load(modelPath), data, "label", "prediction", debug)
          .select("sizeReset","prediction")
        .transform(writeCsv(spark, input, output))
      case "LOGISTIC_REGRESSION" => Evaluators.evaluateLinearRegression(LogisticRegressionModel.load(modelPath), data, "label", "prediction").show(100)
      case "DECISION_TREE" => Evaluators.evaluateDecisionTree(DecisionTreeClassificationModel.load(modelPath), data, "label", "prediction").show(100)
      case default => println(s"Unexpected model: $default, please use one of RANDOM_FOREST, LOGISTIC_REGRESSION or DECISION_TREE")
    }
  }

  def writeCsv(spark: SparkSession, input: String, output: String)(dataFrame: DataFrame) = {
    val data: DataFrame = spark.read.format("libsvm").json(input)

    val data1: DataFrame =  data.withColumn("rowId1", monotonically_increasing_id())
    val dataFrame1 =  dataFrame.withColumn("rowId2", monotonically_increasing_id())
    if(!data.columns.contains("label")){
      dataFrame1.withColumn("label",when(col("prediction") === 1,true).otherwise(false))
    }
    val df = dataFrame1.as("df2").join(data1.as("df1"), data1("rowId1") === dataFrame1("rowId2"), "inner").drop("rowId1").drop("rowId2")
    df
      .drop("size")
      .drop("prediction")
      .coalesce(1)
      .write
      .mode("overwrite")
      .format("com.databricks.spark.csv")
      .option("header", value = true)
      .option("delimiter", ",")
      .save(output)
    df
  }
}
