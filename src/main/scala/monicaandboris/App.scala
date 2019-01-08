package monicaandboris

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.HashingTF

/**
 * @author ${user.name}
 */
object App {

  var hashingNumFeatures = Map(
    "UniqueCarrier" -> 400, // 20^2
    "FlightNum" -> 7596*20,
    "TailNum" -> 5506*20,
    "Origin" -> 92416, // 304^2
    "Dest" -> 96100 // 310^2
  )

  def main(args: Array[String]): Unit = {
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Flight delay prediction")
      .getOrCreate

    import spark.implicits._

    val df = spark.read
      .format("csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .load(args(0))

    val strippedDf = df
      // drop forbidden columns
      .drop("ArrTime")
      .drop("ActualElapsedTime")
      .drop("AirTime")
      .drop("TaxiIn")
      .drop("Diverted")
      .drop("CarrierDelay")
      .drop("WeatherDelay")
      .drop("NASDelay")
      .drop("SecurityDelay")
      .drop("LateAircraftDelay")
      // filter out NA rows
      .filter($"ArrDelay" =!= "NA")
      .filter($"DepTime" =!= "NA")
      .filter($"CRSDepTime" =!= "NA")
      .filter($"CRSArrTime" =!= "NA")

    // hashing of nominal variables
    // hashing space based on https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087
    // WIP!
//    val hashingFeatures = hashingNumFeatures.keySet.map{ feature =>
//      new HashingTF()
//        .setNumFeatures(hashingNumFeatures(feature))
//        .setInputCol(feature)
//        .setOutputCol(feature + "_Hash")
//    }.toArray

    // split features with the hhmm format into two columns
    val timeFeatures = List("DepTime", "CRSDepTime", "CRSArrTime").map{ feature =>
      new TimeSplitter().setInputCol(feature)
    }.toArray

    var pipeline = new Pipeline().setStages(timeFeatures)

    val indexer_model = pipeline.fit(strippedDf)
    val df_transformed = indexer_model.transform(strippedDf)


    df_transformed
      .write.format("csv")
      .option("header", "true")
      .mode("overwrite")
      .save(args(1))

  }

}
