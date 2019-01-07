package monicaandboris

/**
 * @author ${user.name}
 */
object ValueCount {


  def main(args: Array[String]): Unit = {
    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Flight delay prediction")
      .getOrCreate

    val df = spark.read
      .format("csv")
      .option("header", "true") //reading the headers
      .option("mode", "DROPMALFORMED")
      .load(args(0))

    val field = df.select(df(args(1))).distinct
    print(field.count())
    print("------------------------------------------------------------------")
//    print(field.show)

  }

}
