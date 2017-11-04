import org.apache.spark.sql.types._

object Constants {

  val schema = StructType(Array(
    StructField("appOrSite", StringType, true),
    StructField("bidFloor", StringType, true),
    StructField("city", StringType, true),
    StructField("exchange", StringType, true),
    StructField("impid", StringType, true),
    StructField("interests", StringType, true),
    StructField("label", DoubleType, false),
    StructField("media", StringType, true),
    StructField("network", StringType, true),
    StructField("os", StringType, true),
    StructField("publisher", StringType, true),
    StructField("size", IntegerType, true),
    StructField("timestamp", IntegerType, true),
    StructField("type", StringType, true),
    StructField("user", StringType, true)
  ))
}