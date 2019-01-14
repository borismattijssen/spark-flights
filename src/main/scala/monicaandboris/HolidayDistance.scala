package monicaandboris

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{DateType, IntegerType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import java.util.{Calendar, Date, GregorianCalendar}
import java.util.concurrent.TimeUnit

class HolidayDistance(override val uid: String) extends Transformer {
  final val inputCol = new Param[String](this, "inputCol", "The input column")
  final val outputCol = new Param[String](this, "outputCol", "The output column")

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def this() = this(Identifiable.randomUID("holidaydistance"))

  def copy(extra: ParamMap): HolidayDistance = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    val idx = schema.fieldIndex($(inputCol))
    val field = schema.fields(idx)
    if (field.dataType != DateType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type DateType")
    }
    // Add the return field
    schema.add($(outputCol), IntegerType, false)
  }

  /*
  New Year’s Day (1st January)
  Martin Luther King’s Day (3rd Monday of January)
  President’s Day (3 Monday of February)
  Memorial Day (Final Monday of May)
  Independence Day: (July 4th)
  Labor Day: (First Monday of September)
  Columbus Day: (Second Monday of October)
  Veterans Day: (November 11th)
  Thanksgiving: (Final Thursday of November)
  Christmas: (December 25th)
   */

  def transform(df: Dataset[_]): DataFrame = {
    val distance = udf { in: java.sql.Date =>
      val calendar = new GregorianCalendar
      calendar.setTime(in)
      val y = calendar.get(Calendar.YEAR)
      val holidays = List(
        Holidays.NewYearsDay(y),
        Holidays.MartinLutherKingObserved(y),
        Holidays.PresidentsDayObserved(y),
        Holidays.MemorialDayObserved(y),
        Holidays.IndependenceDay(y),
        Holidays.LaborDayObserved(y),
        Holidays.ColumbusDayObserved(y),
        Holidays.VeteransDayObserved(y),
        Holidays.ThanksgivingObserved(y),
        Holidays.ChristmasDay(y)
      )
      holidays.map(h => getDateDiff(h.getTime(), in, TimeUnit.DAYS)).min
    }

    df.withColumn($(outputCol), distance(df($(inputCol))))
  }

  /**
    * Get a diff between two dates
    *
    * @param date1    the oldest date
    * @param date2    the newest date
    * @param timeUnit the unit in which you want the diff
    * @return the diff value, in the provided unit
    */
  def getDateDiff(date1: Date, date2: Date, timeUnit: TimeUnit): Long = {
    val diffInMillies = Math.abs(date2.getTime - date1.getTime)
    timeUnit.convert(diffInMillies, TimeUnit.MILLISECONDS)
  }
}
