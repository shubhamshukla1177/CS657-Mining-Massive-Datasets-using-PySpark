# -*- coding: utf-8 -*-

from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import *
from pyspark.sql import *
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
import re
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
spark = SparkSession.builder.appName("BFRSS").getOrCreate()

!pip install pyspark

# df_2011=spark.read.csv('/user/kfnu/2011.csv',header=True,inferSchema=True)
# df_2013=spark.read.csv('/user/kfnu/2013.csv',header=True,inferSchema=True)
# df_2014=spark.read.csv('/user/kfnu/2014.csv',header=True,inferSchema=True)
# df_2015=spark.read.csv('/user/kfnu/2015.csv',header=True,inferSchema=True)
# df_2016=spark.read.csv('/user/kfnu/2016.csv',header=True,inferSchema=True)
# df_2017=spark.read.csv('/user/kfnu/2017.csv',header=True,inferSchema=True)
# df_2018=spark.read.csv('/user/kfnu/2018.csv',header=True,inferSchema=True)



df_2011=spark.read.csv('/kaggle/input/behavioral-risk-factor-surveillance-system/2011.csv',header=True,inferSchema=True)
df_2012=spark.read.csv('/kaggle/input/behavioral-risk-factor-surveillance-system/2012.csv',header=True,inferSchema=True)
df_2013=spark.read.csv('/kaggle/input/behavioral-risk-factor-surveillance-system/2013.csv',header=True,inferSchema=True)
df_2014=spark.read.csv('/kaggle/input/behavioral-risk-factor-surveillance-system/2014.csv',header=True,inferSchema=True)
df_2015=spark.read.csv('/kaggle/input/behavioral-risk-factor-surveillance-system/2015.csv',header=True,inferSchema=True)
df_2016=spark.read.csv('/kaggle/input/extended-behavioural-dataset/2016.csv',header=True,inferSchema=True)
df_2017=spark.read.csv('/kaggle/input/extended-behavioural-dataset/2017.csv',header=True,inferSchema=True)
df_2018=spark.read.csv('/kaggle/input/extended-behavioural-dataset/2018.csv',header=True,inferSchema=True)

df_2011_renamed = df_2011.withColumnRenamed("CIINTFER","DIFFALON").withColumnRenamed('CHCKIDNY','KIDNEY').withColumn("YEAR",lit(2011))

df_2013_renamed = df_2013.withColumnRenamed("DECIDE","CIMEMLOS").withColumnRenamed('CHCCOPD1','CHCCOPD').withColumnRenamed('CHCKIDNY','KIDNEY').withColumn("YEAR",lit(2013)) \
                     .withColumnRenamed('_PACAT1','_PACAT').withColumnRenamed('_PRACE1','_PRACE').withColumn("_RFCHOL",lit(9))

df_2014_renamed = df_2014.withColumnRenamed('CHCCOPD1','CHCCOPD').withColumnRenamed('CHCKIDNY','KIDNEY').withColumnRenamed("DECIDE","CIMEMLOS").withColumn("YEAR",lit(2014)).withColumn("_RFHYPE5",lit(9))  \
                    .withColumnRenamed('_PRACE1','_PRACE').withColumn("_CHOLCHK",lit(9)).withColumn("_RFCHOL",lit(9)).withColumn("_PACAT",lit(9))

df_2015_renamed = df_2015.withColumnRenamed('CHCCOPD1','CHCCOPD').withColumnRenamed('CHCKIDNY','KIDNEY').withColumnRenamed('_RFDRHV5','_RFDRHV4').withColumn("YEAR",lit(2015)) \
                    .withColumnRenamed('_PACAT1','_PACAT').withColumnRenamed('_PRACE1','_PRACE')

df_2016_renamed = df_2016.withColumnRenamed('CHCCOPD1','CHCCOPD').withColumnRenamed('CHCKIDNY','KIDNEY').withColumnRenamed('_RFDRHV5','_RFDRHV4').withColumn("YEAR",lit(2016)) \
                    .withColumnRenamed('_PRACE1','_PRACE').withColumn("_CHOLCHK",lit(9)).withColumn("_RFCHOL",lit(9)).withColumn("_PACAT",lit(9)).withColumn("_RFHYPE5",lit(9))

df_2017_renamed = df_2017.withColumnRenamed('CHCCOPD1','CHCCOPD').withColumnRenamed('CHCKIDNY','KIDNEY').withColumnRenamed('_RFDRHV5','_RFDRHV4').withColumn("YEAR",lit(2017)) \
                    .withColumnRenamed('_PACAT1','_PACAT').withColumnRenamed('_PRACE1','_PRACE').withColumnRenamed("_CHOLCH1","_CHOLCHK").withColumnRenamed("_RFCHOL1","_RFCHOL")

df_2018_renamed = df_2018.withColumnRenamed('CHCCOPD1','CHCCOPD').withColumnRenamed('CHCKDNY1','KIDNEY').withColumnRenamed('SEX1','SEX').withColumn("YEAR",lit(2018)).withColumnRenamed('_RFDRHV6','_RFDRHV4') \
                    .withColumn("_CHOLCHK",lit(9)).withColumn("_RFCHOL",lit(9)).withColumn("_PACAT",lit(9)).withColumn("_RFHYPE5",lit(9)).withColumnRenamed('_PRACE1','_PRACE')

col_2011 = df_2011_renamed.columns
col_2013 = df_2013_renamed.columns
col_2014 = df_2014_renamed.columns
col_2015 = df_2015_renamed.columns
col_2016 = df_2016_renamed.columns
col_2017 = df_2017_renamed.columns
col_2018 = df_2018_renamed.columns

set2011 = set(col_2011)
set2013 = set(col_2013)
set2014 = set(col_2014)
set2015 = set(col_2015)
set2016 = set(col_2016)
set2017 = set(col_2017)
set2018 = set(col_2018)

common_cols = set2011.intersection(set2013,set2014,set2015,set2016,set2017,set2018)
final_cols = list(common_cols)
print("Common Columns among all dataframes:\n", final_cols)

select_cols = ['IYEAR','IMONTH','IDAY','_RFDRHV4','CHCCOPD','_PRACE','_PACAT','_RFCHOL','_CHOLCHK','_RFHYPE5','ALCDAY5', '_STATE', 'HIVTSTD3', 'HLTHPLN1', 'DRNKANY5', '_AIDTST3', 'HTM4', 'ASTHMA3', '_BMI5CAT', 'INSULIN',
               'GENHLTH', 'INCOME2', '_HCVU651', 'PREGNANT', 'HTIN4', 'VETERAN3', 'EDUCA', 'DIFFALON','ASTHNOW', 'CVDCRHD4', 'CASTHDX2', 'PHYSHLTH', 'WTKG3', '_AGE65YR', 'CIMEMLOS', 'SMOKE100', '_INCOMG', 'CHKHEMO3', 'MARITAL',
               'DOCTDIAB', 'DROCDY3_', 'POORHLTH', '_CHLDCNT', 'CASTHNO2', 'HEIGHT3', 'DIABEDU', '_ASTHMS1', 'EXERANY2','_CASTHM1', 'HAVARTH3', '_AGE_G', '_DRDXAR1', 'CHCSCNCR', '_RFHLTH', '_BMI5', 'PDIABTST',  '_AGEG5YR',
               'USENOW3', 'MEDCOST', 'CVDINFR4', 'SMOKDAY2', '_RFBMI5', '_EDUCAG', 'CVDSTRK3', 'DIABEYE', 'DRNK3GE5', 'WEIGHT2', 'LASTSMK2', 'DIABAGE2', 'MENTHLTH', 'MAXDRNKS', 'AVEDRNK2', 'PREDIAB1', '_RFSMOK3', '_SMOKER3',
               'CHECKUP1', 'DIABETE3', 'CHCOCNCR', 'BLDSUGAR', '_RFBING5', 'ADDEPEV2', 'STOPSMK2','SEX','KIDNEY']

df_2011_renamed = df_2011_renamed.select(*select_cols)
df_2013_renamed = df_2013_renamed.select(*select_cols)
df_2014_renamed = df_2014_renamed.select(*select_cols)
df_2015_renamed = df_2015_renamed.select(*select_cols)
df_2016_renamed = df_2016_renamed.select(*select_cols)
df_2017_renamed = df_2017_renamed.select(*select_cols)
df_2018_renamed = df_2018_renamed.select(*select_cols)

base_df = df_2011_renamed.unionByName(df_2013_renamed) \
                         .unionByName(df_2014_renamed) \
                         .unionByName(df_2015_renamed) \
                         .unionByName(df_2016_renamed) \
                         .unionByName(df_2017_renamed) \
                         .unionByName(df_2018_renamed)

#base_df.count()

temp_numerical_df = base_df \
            .withColumn("_STATE", col("_STATE").cast("int")) \
            .withColumn("STATE_CODE", when(col("_STATE")==1, "AL").when(col("_STATE")==2, "AK").when(col("_STATE")==4, "AZ").when(col("_STATE")==5, "AR")
                        .when(col("_STATE")==6, "CA").when(col("_STATE")==20, "KS").when(col("_STATE")==34, "NJ").when(col("_STATE")==48, "TX")
                        .when(col("_STATE")==21, "KY").when(col("_STATE")==35, "NM").when(col("_STATE")==49, "UT").when(col("_STATE")==8, "CO")
                        .when(col("_STATE")==22, "LA").when(col("_STATE")==36, "NY").when(col("_STATE")==50, "VT")
                        .when(col("_STATE")==9, "CT").when(col("_STATE")==23, "ME").when(col("_STATE")==37, "NC").when(col("_STATE")==51, "VA")
                        .when(col("_STATE")==10, "DE").when(col("_STATE")==24, "MD").when(col("_STATE")==38, "ND").when(col("_STATE")==66, "GU")
                        .when(col("_STATE")==11, "DC").when(col("_STATE")==25, "MA").when(col("_STATE")==39, "OH").when(col("_STATE")==53, "WA")
                        .when(col("_STATE")==12, "FL").when(col("_STATE")==26, "MI").when(col("_STATE")==40, "OK").when(col("_STATE")==54, "WV")
                        .when(col("_STATE")==13, "GA").when(col("_STATE")==27, "MN").when(col("_STATE")==41, "OR").when(col("_STATE")==55, "WI")
                        .when(col("_STATE")==28, "MS").when(col("_STATE")==42, "PA").when(col("_STATE")==56, "WY")
                        .when(col("_STATE")==15, "HI").when(col("_STATE")==29, "MO")
                        .when(col("_STATE")==16, "ID").when(col("_STATE")==30, "MT").when(col("_STATE")==44, "RI").when(col("_STATE")==72, "PR")
                        .when(col("_STATE")==17, "IL").when(col("_STATE")==31, "NV").when(col("_STATE")==45, "SC").when(col("_STATE")==18, "IN")
                        .when(col("_STATE")==32, "NV").when(col("_STATE")==46, "SD").when(col("_STATE")==19, "IA").when(col("_STATE")==33, "NH")
                        .when(col("_STATE")==47, "TN").otherwise("UNK")) \
            .withColumn("HAS_MICHD", when((col("CVDINFR4")==1) | (col("CVDCRHD4")==1), lit(1)).otherwise(lit(0))) \
            .withColumn("IS_EMPLOYED", when(col("_INCOMG").isin([1,2,3,4,5])==True, lit(1)).otherwise(lit(0))) \
            .withColumn("MENTAL_HLTH", when(col("MENTHLTH").isin([88,77,99])==True,lit(1)).otherwise(lit(0))) \
            .withColumn("PHYSICAL_HLTH", when(col("_RFHLTH").isin([1,2])==True,lit(1)).otherwise(lit(0))) \
            .withColumn("GENDER", when(col("SEX")==1, "MALE").otherwise("FEMALE")) \
            .withColumn("IS_MARRIED", when(col("MARITAL")==1,lit(1)).otherwise(lit(0))) \
            .withColumn("INCOME_GRP", when(col("_INCOMG")==1,"Less than $15000").when(col("_INCOMG")==2,"$15000 to less than $25000").when(col("_INCOMG")==3,"$25000 to less than $35000")
                                 .when(col("_INCOMG")==4,"$35000 to less than $50000").when(col("_INCOMG")==5,"$50000 or more").otherwise("No Income")) \
            .withColumn("GENERAL_HLTH", when(col("GENHLTH").isin([1,2,3,4])==True, lit(1)).otherwise(lit(0))) \
            .withColumn("EDUCATION_CATEGORY", when(col("EDUCA")==2, "Grades 1 through 8")
                                .when(col("EDUCA")==3, "Grades 9 through 11")
                                .when(col("EDUCA")==4, "Grades 12 or High School Graduate")
                                .when(col("EDUCA")==5, "College 1 year to 3 years")
                                .when(col("EDUCA")==6, "College 4 years or more").otherwise("Never attended school")) \
            .withColumn("HAS_HLTHCOVRGE", when(col("_HCVU651")==1, lit(1)).otherwise(lit(0))) \
            .withColumn("HIGH_BP", when(col("_RFHYPE5")==2, lit(1)).otherwise(lit(0))) \
            .withColumn("HIGH_CHOL", when(col("_RFCHOL")==2, lit(1)).otherwise(lit(0))) \
            .withColumn("HAS_ASTHMA", when(col("_CASTHM1")==2, lit(1)).otherwise(lit(0))) \
            .withColumn("HAS_ARTHRITIS", when(col("_DRDXAR1")==1, lit(1)).otherwise(lit(0))) \
            .withColumn("AGE_CATEGORY", when(col("_AGEG5YR")==1, "AGE 18-24").when(col("_AGEG5YR")==1, "AGE 18-24")
                        .when(col("_AGEG5YR")==1, "AGE 18-24").when(col("_AGEG5YR")==2, "AGE 25-29").when(col("_AGEG5YR")==3, "AGE 30-34")
                        .when(col("_AGEG5YR")==4, "AGE 35-39").when(col("_AGEG5YR")==5, "AGE 40-44").when(col("_AGEG5YR")==6, "AGE 45-49")
                        .when(col("_AGEG5YR")==7, "AGE 50-54").when(col("_AGEG5YR")==8, "AGE 55-59").when(col("_AGEG5YR")==9, "AGE 60-64")
                        .when(col("_AGEG5YR")==10, "AGE 65-69").when(col("_AGEG5YR")==11, "AGE 70-74").when(col("_AGEG5YR")==13, "AGE 75-79").otherwise("AGE 80 or Older")) \
            .withColumn("AGE_18_64", when(col("_AGE65YR")==1, lit(1)).otherwise(lit(0))) \
            .withColumn("WEIGHT_CATEGORY", when(col("_BMI5CAT")==1,"Underweight").when(col("_BMI5CAT")==2,"Normal Weight").when(col("_BMI5CAT")==3,"Overweight").otherwise("Obese")) \
            .withColumn("HEAVY_SMOKER", when(col("_SMOKER3").isin([1,2])==True, lit(1)).otherwise(lit(0))) \
            .withColumn("PHYSICALLY_ACTIVE", when(col("_PACAT").isin([1,2,3])==True, lit(1)).otherwise(lit(0))) \
            .withColumn("RACE", when(col("_PRACE")==1,"White").when(col("_PRACE")==2,"African American").when(col("_PRACE")==3,"Alaskan Native")
                        .when(col("_PRACE")==4,"Asian").when(col("_PRACE")==5,"Native Hawaiian").otherwise("Other Race")) \
            .withColumn("HEAVY_DRINKER", when(col("_RFDRHV4")==2, lit(1)).otherwise(lit(0))) \
            .withColumn("HAS_DEPRESSION", when(col("ADDEPEV2")==1, lit(1)).otherwise(lit(0))) \
            .withColumn("HAS_CANCER", when(col("CHCOCNCR")==1, lit(1)).otherwise(lit(0))) \
            .withColumn("HAS_PREDIABETES", when(col("PREDIAB1").isin([1,2]), lit(1)).otherwise(lit(0))) \
            .withColumn("LAST1YR_DRVISIT", when(col("CHECKUP1")==1, lit(1)).otherwise(lit(0))) \
            .withColumn("NODRVISIT_BCOZOFMEDCOST", when(col("MEDCOST")==1, lit(1)).otherwise(lit(0)))

temp_df = base_df \
            .withColumn("_STATE", col("_STATE").cast("int")) \
            .withColumn("STATE_CODE", when(col("_STATE")==1, "AL").when(col("_STATE")==2, "AK").when(col("_STATE")==4, "AZ").when(col("_STATE")==5, "AR")
                        .when(col("_STATE")==6, "CA").when(col("_STATE")==20, "KS").when(col("_STATE")==34, "NJ").when(col("_STATE")==48, "TX")
                        .when(col("_STATE")==21, "KY").when(col("_STATE")==35, "NM").when(col("_STATE")==49, "UT").when(col("_STATE")==8, "CO")
                        .when(col("_STATE")==22, "LA").when(col("_STATE")==36, "NY").when(col("_STATE")==50, "VT")
                        .when(col("_STATE")==9, "CT").when(col("_STATE")==23, "ME").when(col("_STATE")==37, "NC").when(col("_STATE")==51, "VA")
                        .when(col("_STATE")==10, "DE").when(col("_STATE")==24, "MD").when(col("_STATE")==38, "ND").when(col("_STATE")==66, "GU")
                        .when(col("_STATE")==11, "DC").when(col("_STATE")==25, "MA").when(col("_STATE")==39, "OH").when(col("_STATE")==53, "WA")
                        .when(col("_STATE")==12, "FL").when(col("_STATE")==26, "MI").when(col("_STATE")==40, "OK").when(col("_STATE")==54, "WV")
                        .when(col("_STATE")==13, "GA").when(col("_STATE")==27, "MN").when(col("_STATE")==41, "OR").when(col("_STATE")==55, "WI")
                        .when(col("_STATE")==28, "MS").when(col("_STATE")==42, "PA").when(col("_STATE")==56, "WY")
                        .when(col("_STATE")==15, "HI").when(col("_STATE")==29, "MO")
                        .when(col("_STATE")==16, "ID").when(col("_STATE")==30, "MT").when(col("_STATE")==44, "RI").when(col("_STATE")==72, "PR")
                        .when(col("_STATE")==17, "IL").when(col("_STATE")==31, "NV").when(col("_STATE")==45, "SC").when(col("_STATE")==18, "IN")
                        .when(col("_STATE")==32, "NV").when(col("_STATE")==46, "SD").when(col("_STATE")==19, "IA").when(col("_STATE")==33, "NH")
                        .when(col("_STATE")==47, "TN").otherwise("UNK")) \
            .withColumn("HAS_MICHD", when((col("CVDINFR4")==1) | (col("CVDCRHD4")==1), "Yes").otherwise("No")) \
            .withColumn("IS_EMPLOYED", when(col("_INCOMG").isin([1,2,3,4,5])==True, "Yes").otherwise("No")) \
            .withColumn("MENTAL_HLTH", when(col("MENTHLTH").isin([88,77,99])==True,"Yes").otherwise("No")) \
            .withColumn("PHYSICAL_HLTH", when(col("_RFHLTH").isin([1,2])==True,"Yes").otherwise("No")) \
            .withColumn("GENDER", when(col("SEX")==1, "MALE").otherwise("FEMALE")) \
            .withColumn("IS_MARRIED", when(col("MARITAL")==1,"Yes").otherwise("No")) \
            .withColumn("INCOME_GRP", when(col("_INCOMG")==1,"Less than $15000").when(col("_INCOMG")==2,"$15000 to less than $25000").when(col("_INCOMG")==3,"$25000 to less than $35000")
                                 .when(col("_INCOMG")==4,"$35000 to less than $50000").when(col("_INCOMG")==5,"$50000 or more").otherwise("No Income")) \
            .withColumn("GENERAL_HLTH", when(col("GENHLTH").isin([1,2,3,4])==True, "Yes").otherwise("No")) \
            .withColumn("EDUCATION_CATEGORY", when(col("EDUCA")==2, "Grades 1 through 8")
                                .when(col("EDUCA")==3, "Grades 9 through 11")
                                .when(col("EDUCA")==4, "Grades 12 or High School Graduate")
                                .when(col("EDUCA")==5, "College 1 year to 3 years")
                                .when(col("EDUCA")==6, "College 4 years or more").otherwise("Never attended school")) \
            .withColumn("HAS_HLTHCOVRGE", when(col("_HCVU651")==1, "Yes").otherwise("No")) \
            .withColumn("HIGH_BP", when(col("_RFHYPE5")==2, "Yes").otherwise("No")) \
            .withColumn("HIGH_CHOL", when(col("_RFCHOL")==2, "Yes").otherwise("No")) \
            .withColumn("HAS_ASTHMA", when(col("_CASTHM1")==2, "Yes").otherwise("No")) \
            .withColumn("HAS_ARTHRITIS", when(col("_DRDXAR1")==1, "Yes").otherwise("No")) \
            .withColumn("AGE_CATEGORY", when(col("_AGEG5YR")==1, "AGE 18-24").when(col("_AGEG5YR")==1, "AGE 18-24")
                        .when(col("_AGEG5YR")==1, "AGE 18-24").when(col("_AGEG5YR")==2, "AGE 25-29").when(col("_AGEG5YR")==3, "AGE 30-34")
                        .when(col("_AGEG5YR")==4, "AGE 35-39").when(col("_AGEG5YR")==5, "AGE 40-44").when(col("_AGEG5YR")==6, "AGE 45-49")
                        .when(col("_AGEG5YR")==7, "AGE 50-54").when(col("_AGEG5YR")==8, "AGE 55-59").when(col("_AGEG5YR")==9, "AGE 60-64")
                        .when(col("_AGEG5YR")==10, "AGE 65-69").when(col("_AGEG5YR")==11, "AGE 70-74").when(col("_AGEG5YR")==13, "AGE 75-79").otherwise("AGE 80 or Older")) \
            .withColumn("AGE_18_64", when(col("_AGE65YR")==1, "Yes").otherwise("No")) \
            .withColumn("WEIGHT_CATEGORY", when(col("_BMI5CAT")==1,"Underweight").when(col("_BMI5CAT")==2,"Normal Weight").when(col("_BMI5CAT")==3,"Overweight").otherwise("Obese")) \
            .withColumn("HEAVY_SMOKER", when(col("_SMOKER3").isin([1,2])==True, "Yes").otherwise("No")) \
            .withColumn("PHYSICALLY_ACTIVE", when(col("_PACAT").isin([1,2,3])==True, "Yes").otherwise("No")) \
            .withColumn("RACE", when(col("_PRACE")==1,"White").when(col("_PRACE")==2,"African American").when(col("_PRACE")==3,"Alaskan Native")
                        .when(col("_PRACE")==4,"Asian").when(col("_PRACE")==5,"Native Hawaiian").otherwise("Other Race")) \
            .withColumn("HEAVY_DRINKER", when(col("_RFDRHV4")==2, "Yes").otherwise("No")) \
            .withColumn("HAS_DEPRESSION", when(col("ADDEPEV2")==1, "Yes").otherwise("No")) \
            .withColumn("HAS_CANCER", when(col("CHCOCNCR")==1, "Yes").otherwise("No")) \
            .withColumn("HAS_PREDIABETES", when(col("PREDIAB1").isin([1,2]), "Yes").otherwise("No")) \
            .withColumn("LAST1YR_DRVISIT", when(col("CHECKUP1")==1, "Yes").otherwise("No")) \
            .withColumn("NODRVISIT_BCOZOFMEDCOST", when(col("MEDCOST")==1, "Yes").otherwise("No")) \
            .withColumn("DIFFALONE", when(col("DIFFALON")==1, "Yes").otherwise("No")) \
            .withColumn("BADPHYSHLTH_PAST30DAYS", when((col("PHYSHLTH").cast('int').isin([88,77,99])) | (col("PHYSHLTH").isNull()), lit(0)).otherwise(col("PHYSHLTH").cast('int'))) \
            .withColumn("BADMENTHLTH_PAST30DAYS", when((col("MENTHLTH").cast('int').isin([88,77,99])) | (col("MENTHLTH").isNull()), lit(0)).otherwise(col("MENTHLTH").cast('int'))) \
            .withColumn("POORHLTH_PAST30DAYS", when((col("POORHLTH").cast('int').isin([88,77,99])) | (col("POORHLTH").isNull()), lit(0)).otherwise(col("POORHLTH").cast('int'))) \
            .withColumn("SMOKE_FREQ", when(col("SMOKDAY2").isin([1,2]), lit(1)).otherwise(lit(0))) \
            .withColumn("ACG_ALC", when((col("AVEDRNK2")>=1) & (col("AVEDRNK2")<77), col("AVEDRNK2")).otherwise(lit(0))) \
            .withColumn("YEAR", trim(regexp_replace(col("IYEAR"),"b",""))) \
            .withColumn("MONTH", trim(regexp_replace(col("IMONTH"),"b",""))) \
            .withColumn("DAY", trim(regexp_replace(col("IDAY"),"b",""))) \
            .withColumn("YEARMO",concat(col("YEAR"), col("MONTH"))) \
            .withColumn("KIDNEY", when(col("KIDNEY")==1, "Yes").otherwise("No"))

cols=['YEARMO','YEAR','MONTH','DAY','STATE_CODE', 'HAS_MICHD', 'IS_EMPLOYED', 'MENTAL_HLTH', 'PHYSICAL_HLTH', 'GENDER',
      'IS_MARRIED', 'INCOME_GRP', 'GENERAL_HLTH', 'EDUCATION_CATEGORY', 'HAS_HLTHCOVRGE', 'HIGH_BP', 'HIGH_CHOL',
      'HAS_ASTHMA', 'HAS_ARTHRITIS', 'AGE_CATEGORY', 'AGE_18_64', 'WEIGHT_CATEGORY', 'HEAVY_SMOKER', 'PHYSICALLY_ACTIVE',
      'RACE', 'HEAVY_DRINKER', 'HAS_DEPRESSION', 'HAS_CANCER', 'HAS_PREDIABETES',
      'LAST1YR_DRVISIT', 'NODRVISIT_BCOZOFMEDCOST','DIFFALONE','BADPHYSHLTH_PAST30DAYS','BADMENTHLTH_PAST30DAYS',
      'POORHLTH_PAST30DAYS','ACG_ALC','SMOKE_FREQ','ALCDAY5','KIDNEY','_RFBMI5','ALCDAY5', 'AVEDRNK2' ,'DRNK3GE5' ]
eda_df = temp_df.select(*cols).withColumn("YEARMO", trim(regexp_replace(regexp_replace(col("YEARMO"), " ", "0"),"'","")).cast("int"))

eda_df = eda_df.where(" YEARMO NOT IN (201201,201202)").withColumn("CVD_LABEL", when(col("HAS_MICHD")=="Yes",lit(1)).otherwise(lit(0))).drop('YEAR','MONTH','DAY')

print(temp_df.columns)

noncat_ftrs=['BADPHYSHLTH_PAST30DAYS', 'BADMENTHLTH_PAST30DAYS','POORHLTH_PAST30DAYS','SMOKE_FREQ','ACG_ALC','CVD_LABEL']
cat_ftrs=['GENDER','IS_EMPLOYED','PHYSICAL_HLTH','NODRVISIT_BCOZOFMEDCOST','AGE_CATEGORY','WEIGHT_CATEGORY','EDUCATION_CATEGORY','RACE','PHYSICALLY_ACTIVE', 'HAS_HLTHCOVRGE','HEAVY_DRINKER','HEAVY_SMOKER','HAS_DEPRESSION','IS_MARRIED','HAS_PREDIABETES','DIFFALONE']

temp_df = temp_df.withColumn("HCHOL_1",when(col("HIGH_CHOL")=="Yes", lit(1)).otherwise(lit(0)))

temp_total_count = temp_df.count()
eda_total_count = eda_df.count()

hd_df = temp_numerical_df.where("HAS_MICHD=1").groupBy("STATE_CODE").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = hd_df.toPandas()
plt.figure(figsize=(20, 10))
sns.barplot(x="STATE_CODE", y="Percentage", data=percentage_df)
plt.title('Population having heart disease across all U.S States and Territories)')
plt.xlabel('Heart Disease Across All States')
plt.ylabel('Percent (%)')
plt.show()

michd_fl_gender_df = temp_numerical_df.where("HAS_MICHD=1 AND STATE_CODE='FL' ").groupBy("GENDER").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = michd_fl_gender_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="GENDER", y="Percentage", data=percentage_df)
plt.title('Which Sex has more heart disease in Florida?')
plt.xlabel('Gender')
plt.ylabel('Percent (%)')
plt.show()

isflmaleempl_df = temp_numerical_df.where(" HAS_MICHD=1 AND STATE_CODE='FL' AND GENDER='MALE' ").groupBy("IS_EMPLOYED").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = isflmaleempl_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="IS_EMPLOYED", y="Percentage", data=percentage_df)
plt.title('Employment population of male living in Florida with heart disease?')
plt.xlabel('Employment')
plt.ylabel('Percent (%)')
plt.show()

agecatflmale_df = temp_numerical_df.where(" HAS_MICHD=1 AND STATE_CODE='FL' AND GENDER='MALE' AND IS_EMPLOYED=1 ").groupBy("AGE_CATEGORY").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = agecatflmale_df.toPandas()
plt.figure(figsize=(16, 8))
sns.barplot(x="AGE_CATEGORY", y="Percentage", data=percentage_df)
plt.title('Age Category of male living in Florida with heart disease?')
plt.xlabel('Age Category')
plt.ylabel('Percent (%)')
plt.show()

agecatflmale_df = temp_numerical_df.where("HAS_MICHD=1 AND STATE_CODE='FL' AND GENDER='MALE' AND IS_EMPLOYED=1 AND _AGEG5YR IN (9,10,11,12,13) ").groupBy("HIGH_BP").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = agecatflmale_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HIGH_BP", y="Percentage", data=percentage_df)
plt.title('60+ Male Population with high BP')
plt.xlabel('High BP')
plt.ylabel('Percent (%)')
plt.show()

agecatflmale_df = temp_numerical_df.where("HAS_MICHD=1 AND STATE_CODE='FL' AND GENDER='MALE' AND IS_EMPLOYED=1 AND _AGEG5YR IN (9,10,11,12,13) ").groupBy("HAS_PREDIABETES").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = agecatflmale_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HAS_PREDIABETES", y="Percentage", data=percentage_df)
plt.title('60+ Male Population with Pre-Diabetes')
plt.xlabel('Pre Diabetes')
plt.ylabel('Percent (%)')
plt.show()

agecatflmale_df = temp_numerical_df.where("HAS_MICHD=1 AND STATE_CODE='FL' AND GENDER='MALE' AND IS_EMPLOYED=1 AND _AGEG5YR IN (9,10,11,12,13) ").groupBy("HEAVY_SMOKER").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = agecatflmale_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HEAVY_SMOKER", y="Percentage", data=percentage_df)
plt.title('60+ Male Smoking Population')
plt.xlabel('Heavy Smoker')
plt.ylabel('Percent (%)')
plt.show()

agecatflmale_df = temp_numerical_df.where("HAS_MICHD=1 AND STATE_CODE='FL' AND GENDER='MALE' AND IS_EMPLOYED=1 AND _AGEG5YR IN (9,10,11,12,13) ").groupBy("HIGH_CHOL").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = agecatflmale_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HIGH_CHOL", y="Percentage", data=percentage_df)
plt.title('Population who may get heart attack again?')
plt.xlabel('High Cholesterol')
plt.ylabel('Percent (%)')
plt.show()

humnahisudhrenge = temp_numerical_df.where("HAS_MICHD=1 AND STATE_CODE='FL' AND GENDER='MALE' AND IS_EMPLOYED=1 AND _AGEG5YR IN (9,10,11,12,13) ").groupBy("HEAVY_DRINKER").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = humnahisudhrenge.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HEAVY_DRINKER", y="Percentage", data=percentage_df)
plt.title('Population who may get liver problem in future')
plt.xlabel('High Alcohol Consumption')
plt.ylabel('Percent (%)')
plt.show()

humnahisudhrenge = temp_numerical_df.where("HAS_MICHD=1 AND STATE_CODE='FL' AND GENDER='MALE' AND IS_EMPLOYED=1 AND _AGEG5YR IN (9,10,11,12,13) AND PHYSICALLY_ACTIVE=1 ").groupBy("HEAVY_DRINKER").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = humnahisudhrenge.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HEAVY_DRINKER", y="Percentage", data=percentage_df)
plt.title('Population who may get liver problem in future')
plt.xlabel('High Alcohol Consumption')
plt.ylabel('Percent (%)')
plt.show()

agecatflmale_df = temp_numerical_df.where("HAS_MICHD=1 AND STATE_CODE='FL' AND GENDER='MALE' AND IS_EMPLOYED=1 AND _AGEG5YR IN (9,10,11,12,13) AND PHYSICALLY_ACTIVE=1 ").groupBy("HIGH_CHOL").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = agecatflmale_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HIGH_CHOL", y="Percentage", data=percentage_df)
plt.title('Population who may get heart attack again?')
plt.xlabel('High Cholesterol')
plt.ylabel('Percent (%)')
plt.show()

wtc_df = temp_numerical_df.where("WEIGHT_CATEGORY='Obese'").groupBy("STATE_CODE").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = wtc_df.toPandas()
plt.figure(figsize=(20, 10))
sns.barplot(x="STATE_CODE", y="Percentage", data=percentage_df)
plt.title('Obese Population Across all U.S States and Territories)')
plt.xlabel('Obese Category Across States')
plt.ylabel('Percent (%)')
plt.show()

heartdstate_df = temp_numerical_df.where("HEAVY_SMOKER=1").groupBy("STATE_CODE").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = heartdstate_df.toPandas()
plt.figure(figsize=(20, 10))
sns.barplot(x="STATE_CODE", y="Percentage", data=percentage_df)
plt.title('Population of heavy smoker across all U.S States and Territories)')
plt.xlabel('Smoke Consumption across States')
plt.ylabel('Percent (%)')
plt.show()

nohlthcov_df = temp_numerical_df.where("HAS_HLTHCOVRGE=0").groupBy("STATE_CODE").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = nohlthcov_df.toPandas()
plt.figure(figsize=(20, 10))
sns.barplot(x="STATE_CODE", y="Percentage", data=percentage_df)
plt.title('Population having no health coverage across all U.S States and Territories)')
plt.xlabel('States')
plt.ylabel('Percent (%)')
plt.show()

state_df = temp_numerical_df.groupBy("STATE_CODE").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = state_df.toPandas()
plt.figure(figsize=(20, 10))
sns.barplot(x="STATE_CODE", y="Percentage", data=percentage_df)
plt.title('Population across all U.S States and Territories)')
plt.xlabel('States')
plt.ylabel('Percent (%)')
plt.show()

agec_df = temp_numerical_df.groupBy("AGE_CATEGORY").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = agec_df.toPandas()
plt.figure(figsize=(16, 8))
sns.barplot(x="AGE_CATEGORY", y="Percentage", data=percentage_df)
plt.title('Age Distribution (Across all U.S States and Territories)')
plt.xlabel('Age Category')
plt.ylabel('Percent (%)')
plt.show()

bmidf = temp_numerical_df.groupBy("WEIGHT_CATEGORY").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = bmidf.toPandas()
plt.figure(figsize=(6, 4))
sns.barplot(x="WEIGHT_CATEGORY", y="Percentage", data=percentage_df)
plt.title('Percentage of adults with weight category (All States and Territories)')
plt.xlabel('Weight Category')
plt.ylabel('Percent (%)')
plt.show()

medcost_df = temp_numerical_df.groupBy("NODRVISIT_BCOZOFMEDCOST").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = medcost_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="NODRVISIT_BCOZOFMEDCOST", y="Percentage", data=percentage_df)
plt.title('Percentage of adults who hasnt visited doctor because of cost  (All States and Territories)')
plt.xlabel('No Doctor Visit Because of Cost')
plt.ylabel('Percent (%)')
plt.show()

phyact_df = temp_numerical_df.groupBy("PHYSICALLY_ACTIVE").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = phyact_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="PHYSICALLY_ACTIVE", y="Percentage", data=percentage_df)
plt.title('Percentage of adults who are physically active  (All States and Territories)')
plt.xlabel('Physically Active')
plt.ylabel('Percent (%)')
plt.show()

prediab_df = temp_numerical_df.groupBy("HAS_PREDIABETES").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = prediab_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HAS_PREDIABETES", y="Percentage", data=percentage_df)
plt.title('Percentage of adults with pre diabetic  (All States and Territories)')
plt.xlabel('Pre-Diabetes')
plt.ylabel('Percent (%)')
plt.show()

hlthcvg_df = temp_numerical_df.groupBy("HAS_HLTHCOVRGE").agg((count("*")/temp_total_count*100).alias("Percentage"))
#smokerdf = temp_numerical_df.groupBy("HEAVY_SMOKER").agg(avg(col("HEAVY_SMOKER")).alias("Percentage"))
percentage_df = hlthcvg_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HAS_HLTHCOVRGE", y="Percentage", data=percentage_df)
plt.title('Percentage of adults with No healthcare coverage  (All States and Territories)')
plt.xlabel('Health Coverage')
plt.ylabel('Percent (%)')
plt.show()

bp_df = temp_numerical_df.groupBy("HIGH_BP").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = bp_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HIGH_BP", y="Percentage", data=percentage_df)
plt.title('Percentage of adults with High Blood Pressure  (All States and Territories)')
plt.xlabel('High Blood Pressure')
plt.ylabel('Percent (%)')
plt.show()

smokerdf = temp_numerical_df.groupBy("HEAVY_SMOKER").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = smokerdf.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HEAVY_SMOKER", y="Percentage", data=percentage_df)
plt.title('Percentage of Adults who smoke more often  (All States and Territories)')
plt.xlabel('Heavy Smoker')
plt.ylabel('Percent (%)')
plt.show()

alc_df1 = temp_numerical_df.groupBy("HEAVY_DRINKER").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = alc_df1.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HEAVY_DRINKER", y="Percentage", data=percentage_df)
plt.title('Percentage of Adults who consume alcohol more often  (All States and Territories)')
plt.xlabel('Heavy Alcohol')
plt.ylabel('Percent (%)')
plt.show()

depr_df=temp_numerical_df.groupBy("HAS_DEPRESSION").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = depr_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HAS_DEPRESSION", y="Percentage", data=percentage_df)
plt.title('Percentage of Adults who has depression (All States and Territories)')
plt.xlabel('Depression')
plt.ylabel('Percent (%)')
plt.show()

chol_df = temp_df.groupBy("HCHOL_1").agg((count("*")/temp_total_count*100).alias("Percentage"))
percentage_df = chol_df.toPandas()
plt.figure(figsize=(4, 4))
sns.barplot(x="HCHOL_1", y="Percentage", data=percentage_df)
plt.title('Percentage of Adults who has high Cholesterol (All States and Territories)')
plt.xlabel('High Cholesterol')
plt.ylabel('Percent (%)')
plt.show()

acl_df = eda_df.where("ALCDAY5 > 200 AND ALCDAY5 < 300").groupBy("STATE_CODE").agg(sum((col("ALCDAY5")) / eda_df.count() * 100).alias("Percentage"))
percentage_df = acl_df.toPandas()
plt.figure(figsize=(16, 6))
sns.barplot(x="STATE_CODE", y="Percentage", data=percentage_df)
plt.title('Percentage of Adults who Consumed Alcohol in the Past 30 Days by State')
plt.xlabel('States')
plt.ylabel('Alcohol Consumption (%)')
plt.show()

"""## RANDOM FOREST"""

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import collect_list, explode
from sklearn.metrics import confusion_matrix, precision_recall_curve

def train_random_forest(dataframe, label_column, feature_columns):

    # Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Create a Random Forest classifier
    rf = RandomForestClassifier(labelCol=label_column, featuresCol="features", numTrees=100, maxBins=91 , probabilityCol="probability" )

    # Create a pipeline with the assembler and the Random Forest classifier
    pipeline = Pipeline(stages=[assembler, rf])
    print('Splitting')

    # Split the data into training and test sets
    (training_data, test_data) = dataframe.randomSplit([0.8, 0.2], seed=42)
    print('Splited')

    # Fit the pipeline to the training data
    model = pipeline.fit(training_data)
    print('Fitting')

    # Make predictions on the test data
    predictions = model.transform(test_data)
    print('Predictions are done')

    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_column, metricName="f1")
    f1 = evaluator_f1.evaluate(predictions)
    print(f"F1 Score: {f1}")

    evaluator_precision_recall = MulticlassClassificationEvaluator(labelCol=label_column, metricName="weightedPrecision")
    precision_recall = evaluator_precision_recall.evaluate(predictions)
    print(f"Weighted Precision-Recall: {precision_recall}")

    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=label_column, metricName="accuracy")
    accuracy = evaluator_accuracy.evaluate(predictions)
    print(f"Accuracy: {accuracy}")


    return predictions

heart_features = ['HIGH_CHOL', 'GENDER', 'HIGH_BP', 'PHYSICALLY_ACTIVE', 'POORHLTH_PAST30DAYS', '_RFBMI5', 'AGE_CATEGORY', 'PHYSICAL_HLTH',  'HEAVY_DRINKER']
heart_label = ['HAS_MICHD']
heart_all=['HAS_MICHD' ,'HIGH_CHOL', 'GENDER', 'HIGH_BP', 'PHYSICALLY_ACTIVE', 'POORHLTH_PAST30DAYS', '_RFBMI5', 'AGE_CATEGORY', 'PHYSICAL_HLTH',  'HEAVY_DRINKER']

drinking_features = ['_RFBMI5', 'GENDER', 'PHYSICALLY_ACTIVE', 'POORHLTH_PAST30DAYS', 'HEAVY_DRINKER', 'AGE_CATEGORY', 'PHYSICAL_HLTH', 'ALCDAY5', 'AVEDRNK2', 'DRNK3GE5', 'HAS_DEPRESSION']
drinking_label = ['KIDNEY']
drinking_all=['KIDNEY','_RFBMI5', 'GENDER', 'PHYSICALLY_ACTIVE', 'POORHLTH_PAST30DAYS', 'HEAVY_DRINKER', 'AGE_CATEGORY', 'PHYSICAL_HLTH', 'ALCDAY5', 'AVEDRNK2', 'DRNK3GE5', 'HAS_DEPRESSION']

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

def stringIndexing(df):
    # Create a StringIndexer for each categorical column
    indexers = [
        StringIndexer(inputCol=column, outputCol=column + "_index", handleInvalid="keep")
        for column in df.columns
    ]

    # Create a pipeline
    pipeline = Pipeline(stages=indexers)
    new_df = pipeline.fit(df).transform(df)

    return new_df

"""# CALCULATING PEARSON COEFICIENT"""

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_pearson_correlation(heart_data_final, feature_columns, label_columns):
    # Combine feature columns into a single vector column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_data = assembler.transform(heart_data_final)

    # Calculate the correlation matrix
    correlation_matrix = Correlation.corr(assembled_data, "features").collect()[0]
    feature_columns_no_suffix = [col.replace('_index', '') for col in feature_columns]

    # Extract the Pearson correlation matrix
    pearson_matrix = correlation_matrix[0].toArray()
  # Plot a single bar graph for each label column
    for label_col in label_columns:
        j = label_columns.index(label_col)  # Get the index of the label column
        pearson_coefficients = [pearson_matrix[i, j] for i in range(len(feature_columns))]

        fig, ax = plt.subplots(figsize=(20, 6))
        bar_width = 0.5
        indices = np.arange(len(feature_columns))

        ax.bar(indices, pearson_coefficients, bar_width, label=f"{label_col}", color='blue')

        ax.set_xlabel('Feature Columns')
        ax.set_ylabel('Pearson Coefficient')
        ax.set_title(f'Pearson Coefficients for Each Feature Column with {label_col}')
        ax.set_xticks(indices)
        ax.set_xticklabels(feature_columns)
        ax.legend()

        plt.show()

"""# **F1 SCORE**"""

def plotF1(predictions,features,label):
    prob_and_labels = predictions.select("probability", label).rdd.map(lambda x: (float(x[0][1]), float(x[1])))
    prob_and_labels_df = prob_and_labels.toDF(["probability", label]).toPandas()

    precision, recall, _ = precision_recall_curve(prob_and_labels_df[label], prob_and_labels_df["probability"])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    return prob_and_labels

"""# ROC CURVE"""

from pyspark.sql.functions import col
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plotRoc(prob_and_labels, labels):
    # Convert PipelinedRDD to PySpark DataFrame
    prob_and_labels_df = prob_and_labels.toDF(["probability", labels])
    prob_and_labels_df.take(2)
    prob_and_labels_df.printSchema()
    # Convert PySpark DataFrame to Pandas DataFrame
    prob_and_labels_pandas = prob_and_labels_df.toPandas()
    print(prob_and_labels_pandas.head(4))    # Extract probability and labels as numpy arrays
    y_true = prob_and_labels_pandas[labels].values
    y_prob = prob_and_labels_pandas["probability"].values

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # Compute AUC
    roc_auc = auc(fpr, tpr)
    print('Roc Computed',type(roc_auc),roc_auc)
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

"""# HEART STROKE PREDICTION"""

heart_df.columns

heart_indexed_data = stringIndexing(heart_df)

heart_indexed_feature_cols = [col + "_index" for col in heart_features]
heart_indexed_label_columns = [col + "_index" for col in heart_label]
total=heart_indexed_label_columns +heart_indexed_feature_cols

heart_final=heart_indexed_data.select(*total)

calculate_pearson_correlation(heart_final,heart_indexed_feature_cols,heart_indexed_label_columns)

heart_predictions = train_random_forest(heart_final, heart_indexed_label_columns[0], heart_indexed_feature_cols)

heart_predictions.take(5)

probDF=plotF1(depression_predictions, heart_indexed_feature_cols,heart_indexed_label_columns[0])

plotRoc(probDF, heart_indexed_label_columns[0])

"""## **KIDNEY FALIURE PREDICTION**"""

drinking_indexed_data = stringIndexing(drinking_df)
drinking_indexed_feature_cols = [col + "_index" for col in drinking_features]
drinking_indexed_label_columns = [col + "_index" for col in drinking_label]
drinking_indexed_all = [col + "_index" for col in drinking_all]

drinking_final=drinking_indexed_data.select(*drinking_indexed_all)

calculate_pearson_correlation(drinking_final,drinking_indexed_feature_cols,drinking_indexed_label_columns)

drinking_predictions = train_random_forest(drinking_final, drinking_indexed_label_columns[0], drinking_indexed_feature_cols)

drink_probDF=plotF1(drinking_predictions, drinking_indexed_feature_cols,drinking_indexed_label_columns[0])

plotRoc(drink_probDF, drinking_indexed_label_columns[0])





