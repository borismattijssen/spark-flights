package monicaandboris

import java.io.{FileOutputStream, ObjectOutputStream}

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.sql.functions._

/**
 * @author ${user.name}
 */
object ModelSelection {

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
      .withColumn("Date", to_date(concat(col("Year"), lit("-"), col("Month"), lit("-"), col("DayofMonth"))))


    // index nominal features
    val indexFeatures = List("UniqueCarrier", "FlightNum", "TailNum", "Origin", "Dest").map{ feature =>
      new StringIndexer()
      .setInputCol(feature)
      .setOutputCol(feature + "_Index")
    }.toArray

    // split features with the hhmm format into two columns
    val timeFeatures = List("DepTime", "CRSDepTime", "CRSArrTime").map{ feature =>
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


    // Create dataset
    val pipeline = new Pipeline().setStages(timeFeatures ++ indexFeatures ++ Array(holidayDistance, assembler))
    val indexer_model = pipeline.fit(strippedDf)
    val ds = indexer_model.transform(strippedDf).select("features", "ArrDelay_Int")

    // Split in train chunks and one test chunk
    var alreadySampled = ds.sample(0)
    val subsets = 1 to 6 map{ _ =>
      val subset = ds.except(alreadySampled).sample(false, 0.001)
      alreadySampled = alreadySampled.union(subset)
      subset
    }
    val train = subsets.take(5)
    val test = subsets.drop(1)(1)

    // Define estimator and validator
    val estimator = new RandomForestRegressor()
      .setLabelCol("ArrDelay_Int")
      .setFeaturesCol("features")

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay_Int")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    // Parameters to search for
    val paramMap = new ParamGridBuilder()
      .addGrid(estimator.maxBins, Array(7596))
      .addGrid(estimator.numTrees, Array(10,20,30))
      .addGrid(estimator.maxDepth, Array(10,50,100))
      .build()

    // Training and testing per training chunk per parameter set
    val results = paramMap.map{ param =>
      val trains = train.map{ train_set =>
          println(param)
          val model = estimator.fit(train_set, param).asInstanceOf[Model[_]]
          val metric = evaluator.evaluate(model.transform(test, param))
          println(param.toString() + " ====> " + metric)
          metric
      }
      trains
    }.map(l => (mean(l), stdDev(l)))

    // Reporting
    println("------------- PARAM MAP ---------------")
    println(paramMap)
    println("------------- RESULTS ---------------")
    println(results)

    // Write results to disk for later analysis
    val oos = new ObjectOutputStream(new FileOutputStream(args(1)))
    oos.writeObject((paramMap, results))
    oos.close

    val oos2 = new ObjectOutputStream(new FileOutputStream(args(2)))
    oos2.writeObject((paramMap.toString, results.toString))
    oos2.close



  }

  import Numeric.Implicits._

  def mean[T: Numeric](xs: Iterable[T]): Double = xs.sum.toDouble / xs.size

  def variance[T: Numeric](xs: Iterable[T]): Double = {
    val avg = mean(xs)

    xs.map(_.toDouble).map(a => math.pow(a - avg, 2)).sum / xs.size
  }

  def stdDev[T: Numeric](xs: Iterable[T]): Double = math.sqrt(variance(xs))

}


