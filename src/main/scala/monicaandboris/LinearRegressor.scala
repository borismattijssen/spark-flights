package monicaandboris

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{FeatureHasher, OneHotEncoderEstimator, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression

/**
 * @author ${user.name}
 */
object LinearRegressor {

  // WIP!!
  // hashing space based on https://booking.ai/dont-be-tricked-by-the-hashing-trick-192a6aae3087
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

    val ohe = new OneHotEncoderEstimator()

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
      // cast strings to int
      .withColumn("Year_Int", 'Year cast "int")
      .withColumn("Month_Int", 'Month cast "int")
      .withColumn("DayofMonth_Int", 'DayofMonth cast "int")
      .withColumn("DayOfWeek_Int", 'DayOfWeek cast "int")
      .withColumn("CRSElapsedTime_Int", 'CRSElapsedTime cast "int")
      .withColumn("DepDelay_Int", 'DepDelay cast "int")
      .withColumn("Distance_Int", 'Distance cast "int")
      .withColumn("TaxiOut_Int", 'TaxiOut cast "int")
      .withColumn("ArrDelay_Int", 'ArrDelay cast "int")

    // hashing of nominal variables
    val hasher = new FeatureHasher()
      .setInputCols("UniqueCarrier", "FlightNum", "TailNum", "Origin", "Dest")
      .setOutputCol("HashFeatures")
      .setNumFeatures(2000)

    // split features with the hhmm format into two columns
    val timeFeatures = List("DepTime", "CRSDepTime", "CRSArrTime").map{ feature =>
      new TimeSplitter().setInputCol(feature)
    }.toArray

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "Year_Int",
        "Month_Int",
        "DayofMonth_Int",
        "DayOfWeek_Int",
        "DepTime_Hours",
        "DepTime_Minutes",
        "CRSDepTime_Hours",
        "CRSDepTime_Minutes",
        "CRSArrTime_Hours",
        "CRSArrTime_Minutes",
        "CRSElapsedTime_Int",
        "DepDelay_Int",
        "Distance_Int",
        "TaxiOut_Int",
        "HashFeatures"
      ))
      .setOutputCol("features")


    val pipeline = new Pipeline().setStages(timeFeatures ++ Array(hasher, assembler))

    val indexer_model = pipeline.fit(strippedDf)
    val ds = indexer_model.transform(strippedDf).select("features", "ArrDelay_Int")
    val splits = ds.randomSplit(Array(0.7, 0.3))
    val lr = new LinearRegression()
      .setMaxIter(10)
      .setElasticNetParam(0.8)
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay_Int")

    val lrModel = lr.fit(splits(0))

    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    val predictions = lrModel.transform(splits(1))
    predictions.select("prediction", "ArrDelay_Int", "features").show(10)

  }

}
