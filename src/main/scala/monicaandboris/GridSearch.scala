package monicaandboris

import java.io.{FileOutputStream, ObjectOutputStream}

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.sql.functions._

/**
  * @author ${user.name}
  */
object GridSearch {

  def main(args: Array[String]): Unit = {
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Flight delay prediction")
      .getOrCreate

    import spark.implicits._

    spark.conf.set("spark.sql.broadcastTimeout", 36000)

    val df = spark.read
      .format("csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .load(args(0))

    val monthUDF = udf { m: Int => Math.sin(Math.PI * m / 6.0) }
    val dayOfMonthUDF = udf { d: Int => Math.sin(Math.PI * d / 15.25) }
    val dayOfWeekUDF = udf { d: Int => Math.sin(Math.PI * d / 3.5) }

    // filter out NA rows
    val cleanDf = df
      .filter($"ArrDelay" =!= "NA")
      .filter($"DepTime" =!= "NA")
      .filter($"CRSDepTime" =!= "NA")
      .filter($"CRSArrTime" =!= "NA")
      .sample(false, 0.01)

    val strippedDf = cleanDf
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
        "Dest_Index"
      ))
      .setOutputCol("features")

    val rf = new RandomForestRegressor()
      .setMaxBins(7596)
      .setLabelCol("ArrDelay_Int")
      .setFeaturesCol("features")

    // Create dataset
    val pipeline = new Pipeline().setStages(timeFeatures ++ indexFeatures ++ Array(holidayDistance, assembler, rf))

    // Parameters to search for
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(10, 20, 30))
      .addGrid(rf.maxDepth, Array(10, 50, 100))
      .build()

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator().setLabelCol("ArrDelay_Int").setMetricName("rmse"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3) // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(strippedDf)

    println("------------- RESULTS ---------------")
    println(cvModel.avgMetrics)
    println(cvModel.bestModel)

    // Write results to disk for later analysis
    val oos = new ObjectOutputStream(new FileOutputStream(args(1)))
    oos.writeObject(cvModel)
    oos.close

  }

  import Numeric.Implicits._

  def mean[T: Numeric](xs: Iterable[T]): Double = xs.sum.toDouble / xs.size

  def variance[T: Numeric](xs: Iterable[T]): Double = {
    val avg = mean(xs)

    xs.map(_.toDouble).map(a => math.pow(a - avg, 2)).sum / xs.size
  }

  def stdDev[T: Numeric](xs: Iterable[T]): Double = math.sqrt(variance(xs))

}


