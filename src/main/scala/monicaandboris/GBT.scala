package monicaandboris

import java.io.{FileOutputStream, ObjectOutputStream}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GBTRegressor}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.{col, concat, lit, to_date, udf}
import org.apache.spark.sql.functions._

object GBT {

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

    val monthUDF = udf { m: Int => Math.sin(Math.PI * m / 6.0) }
    val dayOfMonthUDF = udf { d: Int => Math.sin(Math.PI * d / 15.25) }
    val dayOfWeekUDF = udf { d: Int => Math.sin(Math.PI * d / 3.5) }

    import org.apache.spark.sql.expressions.Window
    val nrOfFlightsFromOrigin = Window.partitionBy("Year", "Month", "DayOfMonth", "Origin") // <-- matches groupBy

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
      .withColumn("Date", to_date(concat(col("Year"), lit("-"), col("Month"), lit("-"), col("DayofMonth"))))
      .withColumn("NrOfFlights", count($"Origin") over nrOfFlightsFromOrigin)

    // index nominal features
    val indexFeatures = List("UniqueCarrier", "FlightNum", "TailNum", "Origin", "Dest").map { feature =>
      new StringIndexer()
        .setInputCol(feature)
        .setOutputCol(feature + "_Index")
        .setHandleInvalid("keep")
    }.toArray

    // split features with the hhmm format into two columns
    val timeFeatures = List("DepTime", "CRSDepTime", "CRSArrTime").map { feature =>
      new TimeSplitter().setInputCol(feature)
    }.toArray

    val holidayDistance = new HolidayDistance()
      .setInputCol("Date")
      .setOutputCol("DistanceToHoliday")

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
        "DistanceToHoliday",
        "UniqueCarrier_Index",
        "FlightNum_Index",
        "TailNum_Index",
        "Origin_Index",
        "Dest_Index",
        "NrOfFlights"
      ))
      .setOutputCol("features")

    val gbt = new GBTRegressor()
      .setMaxIter(10)
      .setLabelCol("ArrDelay_Int")
      .setFeaturesCol("features")
      .setMaxBins(7596)


    val pipeline = new Pipeline().setStages(timeFeatures ++ indexFeatures ++ Array(holidayDistance, assembler, gbt))

    val Array(training, testing) = strippedDf.randomSplit(Array(0.7, 0.3))

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay_Int")
      .setPredictionCol("prediction")
      .setMetricName("rmse")


    val model = pipeline.fit(training)

    // Make predictions.
    val predictions = model.transform(testing)

    // Select example rows to display.
    predictions.select("prediction", "ArrDelay_Int", "features").show(20)

    // Select (prediction, true label) and compute test error.

    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    // Write results to disk for later analysis
    val oos = new ObjectOutputStream(new FileOutputStream(args(1)))
    oos.writeObject(model)
    oos.close

  }


}
