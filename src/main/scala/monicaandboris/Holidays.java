//** Copyright Statement ***************************************************
//The Salmon Open Framework for Internet Applications (SOFIA)
// Copyright (C) 1999 - 2002, Salmon LLC
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License version 2
// as published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
// For more information please visit http://www.salmonllc.com
//** End Copyright Statement ***************************************************

//
//
//    Author: Gregory N. Mirsky
//    Updated: John D. Mitchell
//    Version 1.02
//


package monicaandboris;

import java.util.Calendar;


public class Holidays {

//*********************************
// Miscellaneous other holidays are left as an exercise for the reader.
//
// public static Date QuebecCivicHoliday (int nYear)
// {
//  // 03 January YYYY
// }
//
// public static Date AshWednesday (int nYear)
// {
//  // 42 days before easter...
// }
//
// public static Date PalmSunday (int nYear)
// {
//  // Sunday before Easter Sunday...
// }
//
// public static Date MaundayThursday (int nYear)
// {
//  // Thursday before Easter...
// }
//
//  public static Date RoshHashanah(int nYear)
// {
//  Source: William H. Jefferys, Department of Astronomy, University of
//  Texas Austin, TX 78712
//
//         http://quasar.as.utexas.edu
//
//  First, calculate the Golden Number G. This is fundamental to the
//  calculation of both the date of Easter and the Date of Rosh Hashanah.
//  It is intimately connected with the Metonic Cycle. For any year Y, the
//  Golden Number is defined as
//
//  G = Remainder(Y|19) + 1. Don't forget to add the 1!!!
//
//  The following rules are also due to John Horton Conway, of Princeton
//  University. In the Gregorian year Y of the Common Era, Rosh Hashanah
//  normally falls on September N, where
//
//  N + fraction = {[Y/100] - [Y/400] - 2} +
//  765433/492480*Remainder(12G|19) + Remainder(Y|4)/4 - (313Y+89081)/98496
//
//  Here, G is the Golden Number, and * means multiply. However, if certain
//  conditions are satisfied, Rosh Hashanah is postponed by one or even two
//  days, as follows:
//
//      ***Postponement rules***
//
//  1.If the day calculated above is a Sunday, Wednesday, or Friday, Rosh
//  Hashanah falls on the next day (Monday, Thursday or Saturday,
//  respectively).
//
//  2.If the calculated day is a Monday, and if the fraction is greater
//  than or equal to 23269/25920, and if Remainder(12G|19) is greater than
//  11, Rosh Hashanah falls on the next day, a Tuesday.
//
//  3.If it is a Tuesday, and if the fraction is greater than or equal to
//  1367/2160, and if Remainder(12G|19) is greater than 6, Rosh Hashanah
//  falls two days later, on Thursday (NOT WEDNESDAY!!).
// }
//
//  public static Date Passover(int nYear)
// {
//  Source: William H. Jefferys, Department of Astronomy, University of
//  Texas Austin, TX 78712
//
//         http://quasar.as.utexas.edu
//
//  Once you have determined the date of Rosh Hashanah, it is easy to
//  calculate the date of Passover in the same (Gregorian or Julian)
//  year. Let M = the number of days from September 6 to Rosh Hashanah.
//  In the example for 1996, M=September 14-September 6 = 8 days.
//
//  Count M days from March 27. That is the date of Passover. It actually
//  begins at sundown on the previous evening. In the example for 1996, 8
//  days after March 27 is April 4 (there are 31 days in March), so
//  Passover begins at sundown on April 3.
// }
//
// public static Date DominionDay (int nYear)
// {
//  // 01 July YYYY
// }
//
// public static Date BoxingDay (int nYear)
//  {
//  // Day after Christmas, December 26th...
//  }
//
//*********************************************


    public static java.util.Calendar AbrahamLincolnsBirthday(int nYear) {
        int nMonth = 1; // February
        // February 12th

        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 12);
        return cal;
    }

    public static java.util.Calendar ChristmasDay(int nYear) {
        int nMonth = 11; // Decmeber
        // December 25th
        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 25);
        return cal;
    }

    public static java.util.Calendar ChristmasDayObserved(int nYear) {
        int nX;
        int nMonth = 11; // December
        java.util.Calendar cal;

        cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 25);
        nX = cal.get(Calendar.DAY_OF_WEEK);
        switch (nX) {
            case 0: {// Sunday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 26);
                return cal;
            }
            case 1: // Monday
            case 2: // Tuesday
            case 3: // Wednesday
            case 4: // Thrusday
            case 5: { // Friday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 25);
                return cal;
            }
            default: {
                // Saturday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 24);
                return cal;
            }
        }
    }

    public static java.util.Calendar ColumbusDayObserved(int nYear) {
        // Second Monday in October
        int nX;
        int nMonth = 9; // October
        java.util.Calendar cal;

        cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 1);
        nX = cal.get(Calendar.DAY_OF_WEEK);
        switch (nX) {
            case 0: {// Sunday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 9);
                return cal;
            }
            case 1: {// Monday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 8);
                return cal;
            }
            case 2: // Tuesday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 14);
                return cal;
            }
            case 3: // Wednesday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 13);
                return cal;
            }
            case 4: // Thrusday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 12);
                return cal;
            }
            case 5: // Friday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 11);
                return cal;
            }
            default: // Saturday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 10);
                return cal;
            }
        }

    }

    public static java.util.Calendar EasterMonday(int nYear) {
        int nEasterMonth = 0;
        int nEasterDay = 0;
        int nMonthMarch = 2; // March
        int nMonthApril = 3; // April
        java.util.Calendar cEasterSunday = EasterSunday(nYear);
        nEasterMonth = cEasterSunday.get(Calendar.MONTH);
        nEasterDay = cEasterSunday.get(Calendar.DAY_OF_MONTH);
        if (nEasterMonth == nMonthMarch || nEasterDay == 31) {
            java.util.Calendar cal = java.util.Calendar.getInstance();
            cal.set(nYear, nMonthApril, 1);
            return cal;
        } else {
            java.util.Calendar cal = java.util.Calendar.getInstance();
            cal.set(nYear, nEasterMonth, ++nEasterDay);
            return cal;
        }
    }

    public static java.util.Calendar EasterSunday(int nYear) {
/*  Calculate Easter Sunday

  Written by Gregory N. Mirsky

  Source: 2nd Edition by Peter Duffett-Smith. It was originally from
  Butcher's Ecclesiastical Calendar, published in 1876. This
  algorithm has also been published in the 1922 book General
  Astronomy by Spencer Jones; in The Journal of the British
  Astronomical Association (Vol.88, page 91, December 1977); and in
  Astronomical Algorithms (1991) by Jean Meeus.

  This algorithm holds for any year in the Gregorian Calendar, which
  (of course) means years including and after 1583.

        a=year%19
        b=year/100
        c=year%100
        d=b/4
        e=b%4
        f=(b+8)/25
        g=(b-f+1)/3
        h=(19*a+b-d-g+15)%30
        i=c/4
        k=c%4
        l=(32+2*e+2*i-h-k)%7
        m=(a+11*h+22*l)/451
        Easter Month =(h+l-7*m+114)/31  [3=March, 4=April]
        p=(h+l-7*m+114)%31
        Easter Date=p+1     (date in Easter Month)

  Note: Integer truncation is already factored into the
  calculations. Using higher percision variables will cause
  inaccurate calculations.
*/

        int nA = 0;
        int nB = 0;
        int nC = 0;
        int nD = 0;
        int nE = 0;
        int nF = 0;
        int nG = 0;
        int nH = 0;
        int nI = 0;
        int nK = 0;
        int nL = 0;
        int nM = 0;
        int nP = 0;
        int nEasterMonth = 0;
        int nEasterDay = 0;

        // Calculate Easter
        if (nYear < 1900) {
            // if year is in java format put it into standard
            // format for the calculation
            nYear += 1900;
        }
        nA = nYear % 19;
        nB = nYear / 100;
        nC = nYear % 100;
        nD = nB / 4;
        nE = nB % 4;
        nF = (nB + 8) / 25;
        nG = (nB - nF + 1) / 3;
        nH = (19 * nA + nB - nD - nG + 15) % 30;
        nI = nC / 4;
        nK = nC % 4;
        nL = (32 + 2 * nE + 2 * nI - nH - nK) % 7;
        nM = (nA + 11 * nH + 22 * nL) / 451;

        //  [3=March, 4=April]
        nEasterMonth = (nH + nL - 7 * nM + 114) / 31;
        --nEasterMonth;
        nP = (nH + nL - 7 * nM + 114) % 31;

        // Date in Easter Month.
        nEasterDay = nP + 1;

        // Uncorrect for our earlier correction.
        nYear -= 1900;

        // Populate the date object...
        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nEasterMonth, nEasterDay);
        return cal;
    }

    public static java.util.Calendar GoodFridayObserved(int nYear) {
        // Get Easter Sunday and subtract two days
        int nEasterMonth = 0;
        int nEasterDay = 0;
        int nGoodFridayMonth = 0;
        int nGoodFridayDay = 0;
        java.util.Calendar cEasterSunday;

        cEasterSunday = EasterSunday(nYear);
        nEasterMonth = cEasterSunday.get(Calendar.MONTH);
        nEasterDay = cEasterSunday.get(Calendar.DAY_OF_MONTH);
        if (nEasterDay <= 3 && nEasterMonth == 3) { // Check if <= April 3rd

            switch (nEasterDay) {
                case 3:
                    nGoodFridayMonth = nEasterMonth - 1;
                    nGoodFridayDay = nEasterDay - 2;
                    break;
                case 2:
                    nGoodFridayMonth = nEasterMonth - 1;
                    nGoodFridayDay = 31;
                    break;
                case 1:
                    nGoodFridayMonth = nEasterMonth - 1;
                    nGoodFridayDay = 31;
                    break;
                default:
                    nGoodFridayMonth = nEasterMonth;
                    nGoodFridayDay = nEasterDay - 2;
            }
        } else {
            nGoodFridayMonth = nEasterMonth;
            nGoodFridayDay = nEasterDay - 2;
        }

        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nGoodFridayMonth, nGoodFridayDay);
        return cal;
    }

    public static java.util.Calendar Halloween(int nYear) {
        int nMonth = 9;
        // October 31st

        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 31);
        return cal;
    }

    public static java.util.Calendar IndependenceDay(int nYear) {
        int nMonth = 6; // July
        // July 4th

        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 4);
        return cal;
    }

    public static java.util.Calendar IndependenceDayObserved(int nYear) {
        int nX;
        int nMonth = 6; // July

        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 4);

        nX = cal.get(Calendar.DAY_OF_WEEK);
        switch (nX) {
            case 0: // Sunday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 5);
                return cal;
            case 1: // Monday
            case 2: // Tuesday
            case 3: // Wednesday
            case 4: // Thrusday
            case 5: // Friday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 4);
                return cal;
            default:
                // Saturday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 3);
                return cal;
        }
    }

    public static java.util.Calendar LaborDayObserved(int nYear) {
        // The first Monday in September
        int nX;
        int nMonth = 8; // September
        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 1);

        nX = cal.get(Calendar.DAY_OF_WEEK);

        switch (nX) {
            case 0: // Sunday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 2);
                return cal;
            case 1: // Monday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 1);
                return cal;
            case 2: // Tuesday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 7);
                return cal;
            case 3: // Wednesday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 6);
                return cal;
            case 4: // Thrusday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 5);
                return cal;
            case 5: // Friday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 4);
                return cal;
            default: // Saturday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 3);
                return cal;
        }
    }

    public static java.util.Calendar MartinLutherKingObserved(int nYear) {
        // Third Monday in January
        int nX;
        int nMonth = 0; // January
        java.util.Calendar cal;

        cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 1);
        nX = cal.get(Calendar.DAY_OF_WEEK);

        switch (nX) {
            case 0: {// Sunday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 16);
                return cal;
            }
            case 1: {// Monday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 15);
                return cal;
            }
            case 2: // Tuesday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 21);
                return cal;
            }
            case 3: // Wednesday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 20);
                return cal;
            }
            case 4: // Thrusday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 19);
                return cal;
            }
            case 5: // Friday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 18);
                return cal;
            }
            default: // Saturday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 17);
                return cal;
            }

        }
    }

    public static java.util.Calendar MemorialDayObserved(int nYear) {
        // Last Monday in May
        int nX;
        int nMonth = 4; //May
        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 31);

        nX = cal.get(Calendar.DAY_OF_WEEK);


        switch (nX) {
            case 0: // Sunday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 25);
                return cal;
            case 1: // Monday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 31);
                return cal;
            case 2: // Tuesday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 30);
                return cal;
            case 3: // Wednesday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 29);
                return cal;
            case 4: // Thrusday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 28);
                return cal;
            case 5: // Friday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 27);
                return cal;
            default: // Saturday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 26);
                return cal;

        }
    }

    public static java.util.Calendar NewYearsDay(int nYear) {
        // January 1st
        int nMonth = 0; // January

        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 1);

        return cal;
    }

    public static java.util.Calendar NewYearsDayObserved(int nYear) {
        int nX;
        int nMonth = 0;     // January
        int nMonthDecember = 11;  // December

        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 1);

        nX = cal.get(Calendar.DAY_OF_WEEK);

        if (nYear > 1900)
            nYear -= 1900;

        switch (nX) {
            case 0: // Sunday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 2);
                return cal;
            case 1: // Monday
            case 2: // Tuesday
            case 3: // Wednesday
            case 4: // Thrusday
            case 5: // Friday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 1);
                return cal;
            default:
                // Saturday, then observe on friday of previous year
                cal = java.util.Calendar.getInstance();
                cal.set(--nYear, nMonthDecember, 31);
                return cal;
        }
    }

    public static java.util.Calendar PresidentsDayObserved(int nYear) {
        // Third Monday in February
        int nX;
        int nMonth = 1; // February

        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 1);

        nX = cal.get(Calendar.DAY_OF_WEEK);

        switch (nX) {
            case 0: {// Sunday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 16);
                return cal;
            }
            case 1: {// Monday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 15);
                return cal;
            }
            case 2: // Tuesday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 21);
                return cal;
            }
            case 3: // Wednesday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 20);
                return cal;
            }
            case 4: // Thrusday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 19);
                return cal;
            }
            case 5: // Friday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 18);
                return cal;
            }
            default: // Saturday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 17);
                return cal;
            }
        }
    }

    public static java.util.Calendar ThanksgivingObserved(int nYear) {
        int nX;
        int nMonth = 10; // November
        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 1);

        nX = cal.get(Calendar.DAY_OF_WEEK);
        switch (nX) {

            case 0: {// Sunday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 26);
                return cal;
            }
            case 1: {// Monday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 25);
                return cal;
            }
            case 2: // Tuesday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 24);
                return cal;
            }
            case 3: // Wednesday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 23);
                return cal;
            }
            case 4: // Thrusday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 22);
                return cal;
            }
            case 5: // Friday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 28);
                return cal;
            }
            default: // Saturday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 27);
                return cal;
            }
        }
    }

    public static java.util.Calendar USElectionDay(int nYear) {
        // First Tuesday in November
        int nX;
        int nMonth = 10; // November
        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 1);
        nX = cal.get(Calendar.DAY_OF_WEEK);
        switch (nX) {
            case 0: {// Sunday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 3);
                return cal;
            }
            case 1: {// Monday
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 2);
                return cal;
            }
            case 2: // Tuesday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 1);
                return cal;
            }
            case 3: // Wednesday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 7);
                return cal;
            }
            case 4: // Thrusday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 6);
                return cal;
            }
            case 5: // Friday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 5);
                return cal;
            }
            default: // Saturday
            {
                cal = java.util.Calendar.getInstance();
                cal.set(nYear, nMonth, 4);
                return cal;
            }
        }
    }

    public static java.util.Calendar ValentinesDay(int nYear) {
        int nMonth = 1; // February
        // February 14th

        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 14);
        return cal;
    }

    public static java.util.Calendar VeteransDayObserved(int nYear) {
        //November 11th
        int nMonth = 10; // November
        java.util.Calendar cal = java.util.Calendar.getInstance();
        cal.set(nYear, nMonth, 11);
        return cal;
    }

    public static String getClassInfo() {
        return ("Name: Holidays\r\n" +
                "Author: Gregory N. Mirsky\r\n" +
                "Updated: John D. Mitchell\r\n" +
                "Version 1.02\r\n" +
                "Copyright 1997, All rights reserved.");
    }
}
