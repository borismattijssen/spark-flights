GBT with Holidays and NrOfFlights RMSE on test: 10.86
GBT without Holidays and NrOfFloghts RMSE on test: 10.897049947861362

-- `` ` --
GBT with Holidays, without NrOfFlights, RMSE on test data = 10.933632365003108
Feature importances
  (20, [10, 11, 13, 16, 17, 18, 19], [0.03673811599210462, 0.23440569939156847, 0.2389036564374425, 0.23664289500311758, 0.10704790155786204, 0.14385288068366206, 0.0024088509342427003] )
"CRSElapsedTime_Int",
"DepDelay_Int",
"TaxiOut_Int",
"FlightNum_Index",
"TailNum_Index",
"Origin_Index",
"Dest_Index"
training time: 3796.549520171 seconds

  -- `` ---
GBT no Holidays, with NrOfFlights
Training time: 4280.719032133 seconds
  +-------------------+------------+--------------------+
| prediction | ArrDelay_Int | features |
+-------------------+------------+--------------------+
| 3.816248758096189 | 13 |[2007.0, 0.4999999...|
| 15.619197208049085 | 24 |[2007.0, 0.4999999...|
| 86.76109974699017 | 80 |[2007.0, 0.4999999...|
| 2.909867411584474 | 2 |[2007.0, 0.4999999...|
| - 6.453149685704963 | -16 |[2007.0, 0.4999999...|
| - 6.352169317510178 | -18 |[2007.0, 0.4999999...|
| 2.9833625170189695 | 3 |[2007.0, 0.4999999...|
| - 6.74968635054805 | 4 |[2007.0, 0.4999999...|
| - 6.825970443844296 | -3 |[2007.0, 0.4999999...|
| - 17.32062309702564 | -22 |[2007.0, 0.4999999...|
| - 6.152882773155295 | -13 |[2007.0, 0.4999999...|
| - 6.16803687203155 | -5 |[2007.0, 0.4999999...|
|- 11.049202791955056 | -6 |[2007.0, 0.4999999...|
| - 5.664782696115512 | -17 |[2007.0, 0.4999999...|
| - 6.182121205645098 | 0 |[2007.0, 0.4999999...|
| - 4.759060097881223 | 4 |[2007.0, 0.4999999...|
| 12.765644271787378 | 17 |[2007.0, 0.4999999...|
| - 6.73785016626502 | -5 |[2007.0, 0.4999999...|
| 1.6932436481586473 | 6 |[2007.0, 0.4999999...|
| - 8.738861538702137 | -9 |[2007.0, 0.4999999...|
+-------------------+------------+--------------------+
only showing top 20 rows
  Root Mean Squared Error (RMSE) on test data = 10.82767821055743
Feature importances
  (20, [1, 10, 11, 13, 15, 16, 17, 18], [0.0012946319396268922, 0.03435905606060148, 0.27642643795828953, 0.23583256959128063, 0.21746975350030273, 0.1019697485809176, 0.12972843294480324, 0.002919369424177782] )
GBTRegressionModel(uid = gbtr_1d5209dfcae5) with 10 trees

"Month_Int",
"CRSElapsedTime_Int",
"DepDelay_Int",
"TaxiOut_Int",
"FlightNum_Index",
"TailNum_Index",
"Origin_Index",
"Dest_Index",

-- `` ---
GBT no Holidays, no Distance
  Training time: 4012.323873856 seconds
  +-------------------+------------+--------------------+
| prediction | ArrDelay_Int | features |
+-------------------+------------+--------------------+
| 4.618715917569705 | 13 |[2007.0, 0.4999999...|
| 52.698250942611715 | 55 |[2007.0, 0.4999999...|
| - 5.630530502781862 | -18 |[2007.0, 0.4999999...|
| - 6.829258847958063 | -6 |[2007.0, 0.4999999...|
| - 2.702528858575592 | -3 |[2007.0, 0.4999999...|
| - 7.134435182864015 | -10 |[2007.0, 0.4999999...|
| - 7.031737297648574 | -3 |[2007.0, 0.4999999...|
|- 10.536798847554476 | -12 |[2007.0, 0.4999999...|
| - 7.58473607551896 | -13 |[2007.0, 0.4999999...|
| 16.17491330550874 | 44 |[2007.0, 0.4999999...|
| - 16.02191505786368 | -22 |[2007.0, 0.4999999...|
| - 6.314857664290364 | -5 |[2007.0, 0.4999999...|
| 27.297940389650954 | 56 |[2007.0, 0.4999999...|
| - 5.877790017447433 | 0 |[2007.0, 0.4999999...|
| 1.1707389760525464 | -7 |[2007.0, 0.4999999...|
| - 9.029369608848421 | -5 |[2007.0, 0.4999999...|
| - 11.1568827489143 | -25 |[2007.0, 0.4999999...|
| - 6.20213479989166 | 0 |[2007.0, 0.4999999...|
| - 7.663430592700249 | -5 |[2007.0, 0.4999999...|
| - 17.08168752521716 | -16 |[2007.0, 0.4999999...|
+-------------------+------------+--------------------+
only showing top 20 rows

Root Mean Squared Error (RMSE) on test data = 10.87092055531735
Feature importances
  (19, [1, 2, 10, 11, 12, 14, 15, 16], [6.172189367518835E-4, 0.0016298999332432882, 0.03395172874703848, 0.25443046002925523, 0.2466312238566643, 0.2209960068611289, 0.10844063123921273, 0.13330283039670526] )

"Month_Int",
"DayofMonth_Int",
"CRSElapsedTime_Int",
"DepDelay_Int",
"TaxiOut_Int",
"FlightNum_Index",
"TailNum_Index",
"Origin_Index",
