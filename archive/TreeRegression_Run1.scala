package monicaandboris

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{FeatureHasher, HashingTF, OneHotEncoderEstimator, VectorAssembler}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, LinearRegression}
import org.apache.spark.sql.functions.udf

/**
 * @author ${user.name}
 */
object TreeRegression {

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

    val monthUDF = udf{m: Int => Math.sin(Math.PI * m / 6.0)}
    val dayOfMonthUDF = udf{d: Int => Math.sin(Math.PI * d / 15.25)}
    val dayOfWeekUDF = udf{d: Int => Math.sin(Math.PI * d / 3.5)}

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
      .withColumn("Month_Int", monthUDF('Month cast "int"))
      .withColumn("DayofMonth_Int", dayOfMonthUDF('DayofMonth cast "int"))
      .withColumn("DayOfWeek_Int", dayOfWeekUDF('DayOfWeek cast "int"))
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

    val dt = new DecisionTreeRegressor()
      .setLabelCol("ArrDelay_Int")
      .setFeaturesCol("features")

    val model = dt.fit(splits(0))
    println("Feature importances")
    println(model.featureImportances)

    // Make predictions.
    val predictions = model.transform(splits(1))

    // Select example rows to display.
    predictions.select("prediction", "ArrDelay_Int", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay_Int")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

  }

}


/*
+ Runtime: 1h7m
+ Best features: [4,6,8,11,13,511,1591]: DepTime_Hours, CRSDepTime_Hours, CRSArrTime_Hours, DepDelay_Int, TaxiOut_Int, two hash features
+ RMSE 19.01763199446498
*/