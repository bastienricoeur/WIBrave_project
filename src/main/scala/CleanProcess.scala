import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.DataFrame

object CleanProcess {
  def os(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("os", when(((!(lower(col("os")) === "ios") && !(lower(col("os")) === "android")) || col("os").isNull), "other").otherwise(lower(col("os"))))
  }

  def label(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("label", when(col("label") === true, 1.0).otherwise(0.0))
  }

  def bidFloor(dataFrame: DataFrame): DataFrame = {
    dataFrame
      .withColumn("bidFloor", when(col("bidFloor") < 2,"<2").otherwise(col("bidFloor")))
      .withColumn("bidFloor", when(col("bidFloor") >= 2 && col("bidFloor") <4,">=2 & <4").otherwise(col("bidFloor")))
      .withColumn("bidFloor", when(col("bidFloor") >= 4 && col("bidFloor") <10,">=4 & <10").otherwise(col("bidFloor")))
      .withColumn("bidFloor", when(col("bidFloor") >= 10,">=10").otherwise(col("bidFloor")))
      .withColumn("bidFloor", when(col("bidFloor").isNull,"null").otherwise(col("bidFloor")))
  }

  def timestamp(dataFrame: DataFrame): DataFrame = {
    dataFrame
      .withColumn("timestamp",col("timestamp").cast(TimestampType))
      .withColumn("timestamp",hour(col("timestamp")))
  }

  def sizeBanner(dataFrame: DataFrame): DataFrame = {
    dataFrame
      .withColumn("sizeReset",when(!col("size").isNull,"notNull").otherwise("null"))
      .withColumn("sizeReset", when(col("size")(0)<200 && col("size")(1)<200,"small").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(0)>200,"large").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(1)>200,"height").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(0)>200 && col("size")(1)>200,"medium").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(0)>300,"mediumlarge").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(1)>300,"mediumheight").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(0)>300 && col("size")(1)>300,"mediumBig").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(0)>400,"littlebiglarge").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(1)>400,"littlebigheight").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(0)>400 && col("size")(1)>400,"littleBig").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(0)>500,"biglarge").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(1)>500,"bigheight").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(0)>500 && col("size")(1)>500,"big").otherwise(col("sizeReset")))
      .withColumn("sizeReset", when(col("size")(0) === 320 && col("size")(1) === 480,"320*480").otherwise(col("sizeReset")))
  }

  def interests(dataFrame: DataFrame): DataFrame = {
    dataFrame
      .withColumn("C1", when(col("interests").contains("IAB1-") || col("interests").contains("IAB1,") || col("interests") == "IAB1",1.0).otherwise(0.0))
      .withColumn("C2", when(col("interests").contains("IAB2-") || col("interests").contains("IAB2,") || col("interests") == "IAB2",1.0).otherwise(0.0))
      .withColumn("C3", when(col("interests").contains("IAB3"),1.0).otherwise(0.0))
      .withColumn("C4", when(col("interests").contains("IAB4"),1.0).otherwise(0.0))
      .withColumn("C5", when(col("interests").contains("IAB5"),1.0).otherwise(0.0))
      .withColumn("C6", when(col("interests").contains("IAB6"),1.0).otherwise(0.0))
      .withColumn("C7", when(col("interests").contains("IAB7"),1.0).otherwise(0.0))
      .withColumn("C8", when(col("interests").contains("IAB8"),1.0).otherwise(0.0))
      .withColumn("C9", when(col("interests").contains("IAB9"),1.0).otherwise(0.0))
      .withColumn("C10", when(col("interests").contains("IAB10"),1.0).otherwise(0.0))
      .withColumn("C11", when(col("interests").contains("IAB11"),1.0).otherwise(0.0))
      .withColumn("C12", when(col("interests").contains("IAB12"),1.0).otherwise(0.0))
      .withColumn("C13", when(col("interests").contains("IAB13"),1.0).otherwise(0.0))
      .withColumn("C14", when(col("interests").contains("IAB14"),1.0).otherwise(0.0))
      .withColumn("C15", when(col("interests").contains("IAB15"),1.0).otherwise(0.0))
      .withColumn("C16", when(col("interests").contains("IAB16"),1.0).otherwise(0.0))
      .withColumn("C17", when(col("interests").contains("IAB17"),1.0).otherwise(0.0))
      .withColumn("C18", when(col("interests").contains("IAB18"),1.0).otherwise(0.0))
      .withColumn("C19", when(col("interests").contains("IAB19"),1.0).otherwise(0.0))
      .withColumn("C20", when(col("interests").contains("IAB20"),1.0).otherwise(0.0))
      .withColumn("C21", when(col("interests").contains("IAB21"),1.0).otherwise(0.0))
      .withColumn("C22", when(col("interests").contains("IAB22"),1.0).otherwise(0.0))
      .withColumn("C23", when(col("interests").contains("IAB23"),1.0).otherwise(0.0))
      .withColumn("C24", when(col("interests").contains("IAB24"),1.0).otherwise(0.0))
      .withColumn("C25", when(col("interests").contains("IAB25"),1.0).otherwise(0.0))
      .withColumn("C26", when(col("interests").contains("IAB26"),1.0).otherwise(0.0))
  }

  def media(dataFrame: DataFrame): DataFrame = {
    dataFrame
      .withColumn("media", when(col("media").isNull,"null").otherwise(col("media")))
  }
}