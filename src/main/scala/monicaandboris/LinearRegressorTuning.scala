package monicaandboris

object LinearRegressorTuning {

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
    val cancellationsByCode = Window.partitionBy("Year", "Month", "DayOfMonth", "Origin", "CancellationCode")


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
      .withColumn("NrOfArrivals", count($"Dest") over nrOfArrivals)
      .withColumn("NrOfCancelledNAS", count($"CancellationCode" === 'C') over cancellationsByCode)
      .withColumn("NrOfCancelledSecurity", count($"CancellationCode" === 'D') over cancellationsByCode)
      .withColumn("NrOfCancelledCarrier", count($"CancellationCode" === 'A') over cancellationsByCode)
      .withColumn("NrOfCancelledWeather", count($"CancellationCode" === 'B') over cancellationsByCode)

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
        "TaxiOut_Int",
        "DistanceToHoliday",
        "UniqueCarrier_Index",
        //"FlightNum_Index",
        //"TailNum_Index",
        "Origin_Index",
        "Dest_Index",
        "NrOfArrivals",
        "NrOfDepartures",
        "NrOfCancelledNAS",
        "NrOfCancelledSecurity",
        "NrOfCancelledCarrier",
        "NrOfCancelledWeather"
      ))
      .setOutputCol("features")

    val lr = new LinearRegression()
      .setLabelCol("ArrDelay_Int")
      .setFeaturesCol("features")

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 1.0))
      .build()

    val pipeline = new Pipeline().setStages(timeFeatures ++ indexFeatures ++ Array(holidayDistance, assembler, lr))

    val Array(training, testing) = strippedDf.randomSplit(Array(0.7, 0.3), seed = 42)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay_Int")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
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

    val pipelineModel: Option[PipelineModel] = model.bestModel match {
      case p: PipelineModel => Some(p)
      case _ => None
    }
    val rm = new RegressionMetrics(holdout.rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Int])))
    println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError))
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")

    val lrModel = pipelineModel.flatMap {
      _.stages.collect { case t: LinearRegressionModel => t }.headOption
    }
    lrModel match {
      case Some(v) => printSummary(v)
      case None => None
    }

  }

  def printSummary(lrModel: LinearRegressionModel): Unit = {
    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

  }

}
