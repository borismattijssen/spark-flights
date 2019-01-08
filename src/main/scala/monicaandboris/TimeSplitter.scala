package monicaandboris

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

import org.apache.spark.sql.functions.udf

class TimeSplitter(override val uid: String) extends Transformer {
  final val inputCol= new Param[String](this, "inputCol", "The input column")

  def setInputCol(value: String): this.type = set(inputCol, value)

  def this() = this(Identifiable.randomUID("trimesplitter"))

  def copy(extra: ParamMap): TimeSplitter = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    val idx = schema.fieldIndex($(inputCol))
    val field = schema.fields(idx)
    if (field.dataType != StringType) {
      throw new Exception(s"Input type ${field.dataType} did not match input type StringType")
    }
    // Add the return field
    schema.add(StructField($(inputCol) + "_Hours", IntegerType, false))
    schema.add(StructField($(inputCol) + "_Minutes", IntegerType, false))
  }

  def transform(df: Dataset[_]): DataFrame = {
    val hours = udf { in: String =>
      var h = "%04d".format(in.toInt).substring(0, 2)
      if(h == "")
        h = "0"
      h.toInt
    }
    val minutes = udf { in: String =>
      var m = "%04d".format(in.toInt).substring(2)
      if(m == "")
        m = "0"
      m.toInt
    }

    df.withColumn($(inputCol) + "_Hours", hours(df($(inputCol))))
      .withColumn($(inputCol) + "_Minutes", minutes(df($(inputCol))))
  }
}
