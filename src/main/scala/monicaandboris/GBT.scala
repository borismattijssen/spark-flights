package monicaandboris

object GBT {

  def main(args: Array[String]): Unit = {
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Flight delay prediction")
      .getOrCreate

    spark.sparkContext.setLogLevel("ERROR")

    val df = spark.read
      .format("csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .load(args(0))

    val monthUDF = udf { m: Int => Math.sin(Math.PI * m / 6.0) }
    val dayOfMonthUDF = udf { d: Int => Math.sin(Math.PI * d / 15.25) }
    val dayOfWeekUDF = udf { d: Int => Math.sin(Math.PI * d / 3.5) }
    val nrOfArrivals = Window.partitionBy("Year", "Month", "DayOfMonth", "Dest")

    val nrOfDepartures = Window.partitionBy("Year", "Month", "DayOfMonth", "Origin")

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
      .withColumn("NrOfDepartures", count($"Origin") over nrOfDepartures)
      .withColumn("NrOfArrivals", count($"Origin") over nrOfArrivals)

    strippedDf.show()
    strippedDf.describe().show()


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
        //"Year_Int",
        "Month_Int",
        "DayofMonth_Int",
        "DayOfWeek_Int",
        "DepTime_Hours",
        //"DepTime_Minutes",
        //"CRSDepTime_Hours",
        //"CRSDepTime_Minutes",
        //"CRSArrTime_Hours",
        //"CRSArrTime_Minutes",
        "CRSElapsedTime_Int",
        "DepDelay_Int",
        //"Distance_Int",
        "TaxiOut_Int",
        "DistanceToHoliday",
        //"UniqueCarrier_Index",
        //"FlightNum_Index",
        //"TailNum_Index",
        //"Origin_Index",
        //"Dest_Index",
        "NrOfArrivals",
        "NrOfDepartures"

      ))
      .setOutputCol("features")

    val gbt = new GBTRegressor()
      .setLabelCol("ArrDelay_Int")
      .setFeaturesCol("features")

    val paramGrid = new ParamGridBuilder()
      .addGrid(gbt.stepSize, Array(0.1, 0.01))
      .addGrid(gbt.maxDepth, Array(5, 7))
      .build()


    val pipeline = new Pipeline().setStages(timeFeatures ++ indexFeatures ++ Array(holidayDistance, assembler, gbt))

    val Array(training, testing) = strippedDf.randomSplit(Array(0.7, 0.3), seed = 42)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay_Int")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline) // the estimator can also just be an individual model rather than a pipeline
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.75)

    val startTime = System.nanoTime()
    val model = tvs.fit(training)
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $elapsedTime seconds")

    // Make predictions.
    val predictions = model.transform(testing)
    // Select example rows to display.
    predictions.select("prediction", "ArrDelay_Int", "features").show(20)
    val holdout = predictions.select("prediction", "ArrDelay_Int")

    // Select (prediction, true label) and compute test error.pyc

    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    // Write predictions to csv for further analysis


    //    val oos = new ObjectOutputStream(new FileOutputStream(args(1)))
    val bestModel = model.bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[GBTRegressionModel]
    println("Best Model")
    println("Feature importances")
    println(bestModel.featureImportances)
    if (bestModel.totalNumNodes < 30) {
      println(bestModel.toDebugString) // Print full model.

    } else {
      println(bestModel) // Print model summary.

    }

    val rm = new RegressionMetrics(holdout.rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Int])))
    println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError))
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")


    // Write results to disk for later analysis


    //    oos.writeChars("RMSE: "+ rmse.toString )
    //    oos.close

  }


}
