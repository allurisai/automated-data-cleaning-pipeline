import re
import os
from statistics import mean
import time
import pandas as pd
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType, FloatType, LongType, StringType, NumericType, TimestampType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, VectorAssembler, Tokenizer, HashingTF, IDF, CountVectorizer
)
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.classification import LogisticRegression, NaiveBayes, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class Helper:
    def __init__(self, datasetName):
        self.datasetName = datasetName
        if "spark" in st.session_state and st.session_state.spark._jsc:
            self.spark = st.session_state.spark
        else:
            self.spark = SparkSession.builder.master("local[*]").appName("DataLoadComparison").getOrCreate()
        
    # Extract new features like year, month, day, hour, and day of week from timestamp columns
    def extractTimestampFeatures(self, sparkDf):
        timestampCols = [field.name for field in sparkDf.schema.fields if isinstance(field.dataType, TimestampType)]
        
        for col in timestampCols:
            sparkDf = sparkDf.withColumn(f"{col}_year", F.year(F.col(col)))
            sparkDf = sparkDf.withColumn(f"{col}_month", F.month(F.col(col)))
            sparkDf = sparkDf.withColumn(f"{col}_day", F.dayofmonth(F.col(col)))
            sparkDf = sparkDf.withColumn(f"{col}_hour", F.hour(F.col(col)))
            sparkDf = sparkDf.withColumn(f"{col}_dow", F.dayofweek(F.col(col)))
            
        return sparkDf

    # Compare and return loading times and dataframes for pandas and pyspark
    def compareLoadingTimes(self, uploadedFile):
        startPandas = time.time()
        pandasDf = pd.read_csv(uploadedFile)
        pandasTime = time.time() - startPandas

        uploadFolder = "UploadedDatasets"
        os.makedirs(uploadFolder, exist_ok=True)

        tempPath = os.path.join(uploadFolder, self.datasetName + ".csv")
        with open(tempPath, "wb") as f:
            f.write(uploadedFile.getbuffer())

        startSpark = time.time()
        sparkDf = self.spark.read.csv(tempPath, header=True, inferSchema=True)
        sparkTime = time.time() - startSpark

        return pandasDf, sparkDf, pandasTime, sparkTime


    # Detect missing values in spark dataframe and summarize them
    def detectMissingValues(self, sparkDf):
        missingCountsExprs = [
            F.count(F.when(F.col(c).isNull() | (F.col(c) == ""), c)).alias(c)
            for c in sparkDf.columns
        ]
        
        missingCountsRow = sparkDf.agg(*missingCountsExprs).collect()[0].asDict()
        missingSummaryDf = (
            pd.DataFrame(list(missingCountsRow.items()), columns=["Column", "Missing Count"])
            .sort_values(by="Missing Count", ascending=False)
            .reset_index(drop=True)
        )
        
        missingCondition = None
        for col in sparkDf.columns:
            cond = F.col(col).isNull() | (F.col(col) == "")
            missingCondition = cond if missingCondition is None else (missingCondition | cond)
        rowsWithMissing = sparkDf.filter(missingCondition)
        rowsWithMissingPd = rowsWithMissing.toPandas()
        return missingSummaryDf, rowsWithMissingPd

    # Detect duplicate rows based on selected columns
    def detectDuplicateRows(self, sparkDf, columnsToCheck):
        grouped = sparkDf.groupBy(columnsToCheck).count().filter("count > 1")
        duplicateRows = grouped.drop("count")
        duplicateRowsPd = duplicateRows.toPandas()
        return columnsToCheck, duplicateRowsPd
    
    def detectSkewness(self, sparkDf, predictingColumn):
        skewSummary = []
        for field in sparkDf.schema.fields:
            col = field.name
            if col == predictingColumn:
                continue
            if isinstance(field.dataType, (IntegerType, FloatType, DoubleType, LongType)):
                skewVal = sparkDf.select(F.skewness(col)).collect()[0][0]
                skewSummary.append({
                    "Column": col, "Type": "Numeric", "Skewness": round(skewVal, 3) if skewVal is not None else None
                })
            elif isinstance(field.dataType, StringType):
                freqDf = sparkDf.groupBy(col).count()
                total = freqDf.agg(F.sum("count")).collect()[0][0]
                probDf = freqDf.withColumn("p", F.col("count") / total)
                entropy = probDf.select(
                    F.sum(-F.col("p") * F.log2(F.col("p"))).alias("entropy")
                ).collect()[0]["entropy"]
                skewSummary.append({
                    "Column": col, "Type": "Categorical", "Skewness (Entropy)": round(entropy, 3) if entropy else 0
                })
        return pd.DataFrame(skewSummary).sort_values(by="Column").reset_index(drop=True)
    
    # Detect numeric and categorical outliers in the dataset
    def detectOutliers(self, sparkDf, predictingColumn):
        totalRows = sparkDf.count()
        outlierSummary = []
        for field in sparkDf.schema.fields:
            col = field.name
            if col == predictingColumn:
                continue
            if isinstance(field.dataType, (IntegerType, FloatType, DoubleType, LongType)):
                quantiles = sparkDf.approxQuantile(col, [0.25, 0.75], 0.05)
                if len(quantiles) < 2:
                    continue
                q1, q3 = quantiles
                iqr = q3 - q1
                lower = q1 - 5 * iqr
                upper = q3 + 5 * iqr
                count = sparkDf.filter((F.col(col) < lower) | (F.col(col) > upper)).count()
                outlierSummary.append({
                    "Column": col, "Type": "Numeric", "Outlier Count": count,
                    "Q1": round(q1, 2), "Q3": round(q3, 2), "Lower Bound": round(lower, 2), "Upper Bound": round(upper, 2)
                })
            elif isinstance(field.dataType, StringType):
                freqDf = sparkDf.groupBy(col).count()
                rareValues = freqDf.filter((F.col("count") / totalRows) < 0.1).count()
                outlierSummary.append({
                    "Column": col, "Type": "Categorical", "Outlier Count": rareValues
                })
        return pd.DataFrame(outlierSummary).sort_values(by="Outlier Count", ascending=False).reset_index(drop=True)

    # Detect correlation between features and target column
    def detectCorrelation(self, sparkDf, predictingColumn):
        workingDf = sparkDf
        featureCols = [col for col in workingDf.columns if col != predictingColumn]
        encodedFeatures = []
        targetEncoded = predictingColumn
        if isinstance(workingDf.schema[predictingColumn].dataType, StringType):
            targetIndexer = StringIndexer(inputCol=predictingColumn, outputCol=predictingColumn + "_index", handleInvalid="skip")
            workingDf = targetIndexer.fit(workingDf).transform(workingDf)
            targetEncoded = predictingColumn + "_index"
        for col in featureCols:
            dtype = workingDf.schema[col].dataType
            if isinstance(dtype, NumericType):
                encodedFeatures.append(col)
            elif isinstance(dtype, StringType):
                uniqueCount = workingDf.select(col).distinct().count()
                if uniqueCount < 100:
                    indexer = StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="skip")
                    workingDf = indexer.fit(workingDf).transform(workingDf)
                    encodedFeatures.append(col + "_index")
        if targetEncoded not in workingDf.columns:
            return pd.DataFrame(columns=["Feature", f"Correlation with {predictingColumn}"])
        data = []
        for col in encodedFeatures:
            if col == targetEncoded:
                continue
            try:
                corr = workingDf.stat.corr(col, targetEncoded)
                if corr is not None:
                    data.append((col.replace("_index", ""), round(corr, 3)))
            except:
                continue
        corrDf = pd.DataFrame(data, columns=["Feature", f"Correlation with {predictingColumn}"])
        corrDf = corrDf.sort_values(by=f"Correlation with {predictingColumn}", ascending=False)
        return corrDf

    # Handle missing values in different ways based on the selected strategy
    def handleMissingValuesSpark(self, sparkDf, strategy):
        if strategy == "Drop Rows":
            return sparkDf.na.drop()
        elif strategy == "Fill with Mean (numeric only)":
            numericCols = [f.name for f in sparkDf.schema.fields if isinstance(f.dataType, (IntegerType, FloatType, DoubleType, LongType))]
            stats = sparkDf.select([mean(col).alias(col) for col in numericCols]).collect()[0].asDict()
            return sparkDf.fillna(stats)
        elif strategy == "Fill with Median (numeric only)":
            numericCols = [f.name for f in sparkDf.schema.fields if isinstance(f.dataType, (IntegerType, FloatType, DoubleType, LongType))]
            medians = {}
            for col in numericCols:
                q = sparkDf.approxQuantile(col, [0.5], 0.01)
                if q:
                    medians[col] = q[0]
            return sparkDf.fillna(medians)
        elif strategy == "Fill with Zero":
            return sparkDf.fillna(0)
        elif strategy == "Fill with 'Unknown'":
            return sparkDf.fillna("Unknown")
        return sparkDf

    # Remove exact duplicate rows in the spark dataframe
    def removeDuplicatesSpark(self, sparkDf):
        print("Columns before dropping duplicates:", sparkDf.columns)
        sparkDf = sparkDf.dropDuplicates()
        print("Columns after dropping duplicates:", sparkDf.columns)
        return sparkDf

    # Detect numeric columns with skewness above the specified threshold
    def detectHighSkewNumeric(self, sparkDf, threshold=1.0):
        skewData = []
        for field in sparkDf.schema.fields:
            if isinstance(field.dataType, NumericType):
                skew = sparkDf.selectExpr(f"skewness({field.name})").first()[0]
                if skew is not None and abs(skew) > threshold:
                    skewData.append((field.name, round(skew, 3)))
        return skewData

    # Fix high skewness in numeric columns by applying transformations like log, sqrt, or cbrt
    def fixHighSkewForRegression(self, sparkDf, method="log", threshold=1.0):
        skewed = self.detectHighSkewNumeric(sparkDf, threshold)
        colsToFix = [col for col, skew in skewed]
        if not colsToFix:
            return sparkDf, []
        for colName in colsToFix:
            if method == "log":
                sparkDf = sparkDf.withColumn(colName, F.log1p(F.col(colName)))
            elif method == "sqrt":
                sparkDf = sparkDf.withColumn(colName, F.sqrt(F.col(colName)))
            elif method == "cbrt":
                sparkDf = sparkDf.withColumn(colName, F.pow(F.col(colName), 1.0/3))
            else:
                raise ValueError(f"Unknown method '{method}', expected one of: log, sqrt, cbrt")
        return sparkDf, colsToFix

    # Remove rows containing numeric outliers based on IQR filtering
    def removeNumericOutliers(self, sparkDf, targetCol):
        numericCols = [
            field.name for field in sparkDf.schema.fields
            if isinstance(field.dataType, NumericType) and field.name != targetCol
        ]
        initialCount = sparkDf.count()
        for colName in numericCols:
            q1, q3 = sparkDf.approxQuantile(colName, [0.25, 0.75], 0.05)
            iqr = q3 - q1
            lower = q1 - 5 * iqr
            upper = q3 + 5 * iqr
            sparkDf = sparkDf.filter((F.col(colName) >= lower) & (F.col(colName) <= upper))
        finalCount = sparkDf.count()
        changedRows = initialCount - finalCount
        return sparkDf, changedRows

    # Clean text columns by lowercasing, removing URLs, and trimming spaces
    def cleanTextColumns(self, sparkDf, colnames):
        def clean(text):
            if text is None:
                return text
            text = text.lower()
            text = re.sub(r"http\S+|www\S+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        cleanUdf = udf(clean, StringType())
        for colname in colnames:
            sparkDf = sparkDf.withColumn(colname, cleanUdf(F.col(colname)))
        return sparkDf

    # Perform random oversampling to balance classes
    def randomOversample(self, sparkDf, targetCol):
        pandasDf = sparkDf.toPandas()
        X = pandasDf.drop(columns=[targetCol])
        y = pandasDf[targetCol]
        ros = RandomOverSampler(random_state=42)
        xResampled, yResampled = ros.fit_resample(X, y)
        resampledDf = xResampled.copy()
        resampledDf[targetCol] = yResampled
        return self.spark.createDataFrame(resampledDf)

    # Perform random undersampling to balance classes
    def undersample(self, sparkDf, targetCol):
        pandasDf = sparkDf.toPandas()
        X = pandasDf.drop(columns=[targetCol])
        y = pandasDf[targetCol]
        rus = RandomUnderSampler(random_state=42)
        xRes, yRes = rus.fit_resample(X, y)
        dfRes = xRes.copy()
        dfRes[targetCol] = yRes
        return self.spark.createDataFrame(dfRes)

    # Train and evaluate multiple regression models and return their RMSE and MSE
    def evaluateRegressionModels(self, sparkDf, target):
        featureCols = [col for col in sparkDf.columns if col != target]
        stages = []
        results = []
        timestampCols = [field.name for field in sparkDf.schema.fields if isinstance(field.dataType, TimestampType)]
        for tsCol in timestampCols:
            sparkDf = sparkDf.withColumn(f"{tsCol}_year", F.year(col(tsCol)))
            sparkDf = sparkDf.withColumn(f"{tsCol}_month", F.month(col(tsCol)))
            sparkDf = sparkDf.withColumn(f"{tsCol}_day", F.dayofmonth(col(tsCol)))
            sparkDf = sparkDf.drop(tsCol)
        sparkDf = sparkDf.filter(F.col(target).isNotNull())
        sparkDf = sparkDf.fillna("missing")
        sparkDf = sparkDf.fillna(-1)
        for colname in featureCols:
            if colname not in sparkDf.columns:
                continue
            dtype = sparkDf.schema[colname].dataType
            if isinstance(dtype, StringType):
                stages.append(StringIndexer(inputCol=colname, outputCol=colname + "_idx", handleInvalid="skip"))
        finalFeatures = [
            c + "_idx" if c in sparkDf.columns and isinstance(sparkDf.schema[c].dataType, StringType)
            else c for c in featureCols if c in sparkDf.columns
        ]
        stages.append(VectorAssembler(inputCols=finalFeatures, outputCol="features"))
        labelCol = target
        if isinstance(sparkDf.schema[target].dataType, StringType):
            labelCol = target + "_idx"
            stages.append(StringIndexer(inputCol=target, outputCol=labelCol, handleInvalid="skip"))
        maxBins = self.getDynamicMaxBins(sparkDf, target)
        models = {
            "Linear Regression": LinearRegression(featuresCol="features", labelCol=labelCol),
            "Decision Tree": DecisionTreeRegressor(featuresCol="features", labelCol=labelCol, maxBins=maxBins),
            "Random Forest": RandomForestRegressor(featuresCol="features", labelCol=labelCol, maxBins=maxBins)
        }
        results = []
        train, test = sparkDf.randomSplit([0.8, 0.2], seed=42)
        for name, model in models.items():
            pipeline = Pipeline(stages=stages + [model])
            try:
                fitted = pipeline.fit(train)
                preds = fitted.transform(test)
                evaluatorRmse = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="rmse")
                evaluatorMse = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="mse")
                rmse = evaluatorRmse.evaluate(preds)
                mse = evaluatorMse.evaluate(preds)
                results.append((name, round(rmse, 3), round(mse, 3)))
            except Exception as e:
                results.append((name, f"Failed: {str(e).splitlines()[0]}"))
        return results

    # Get a dynamic maxBins value based on categorical cardinalities
    def getDynamicMaxBins(self, sparkDf, targetCol):
        maxCardinality = 32
        for field in sparkDf.schema.fields:
            if isinstance(field.dataType, StringType) and field.name != targetCol:
                cardinality = sparkDf.select(field.name).distinct().count()
                if cardinality > maxCardinality:
                    maxCardinality = cardinality
        maxBins = maxCardinality * 2
        return maxBins

    # Train and evaluate multiple classification models and return their performance
    def evaluateClassificationModels(self, sparkDf, target):
        featureCols = [col for col in sparkDf.columns if col != target]
        stages = []
        finalFeatures = []
        timestampCols = [f.name for f in sparkDf.schema.fields if isinstance(f.dataType, TimestampType)]
        for tsCol in timestampCols:
            sparkDf = sparkDf.withColumn(f"{tsCol}_year", F.year(col(tsCol)))
            sparkDf = sparkDf.withColumn(f"{tsCol}_month", F.month(col(tsCol)))
            sparkDf = sparkDf.withColumn(f"{tsCol}_day", F.dayofmonth(col(tsCol)))
            sparkDf = sparkDf.drop(tsCol)
        sparkDf = sparkDf.filter(F.col(target).isNotNull())
        sparkDf = sparkDf.fillna("missing")
        sparkDf = sparkDf.fillna(-1)
        print(f"Rows after dropna: {sparkDf.count()}")
        for colname in featureCols:
            if colname not in sparkDf.columns:
                continue
            dtype = sparkDf.schema[colname].dataType
            if isinstance(dtype, StringType):
                avgWordCount = (
                    sparkDf.select(F.length(F.col(colname)).alias("len"))
                    .agg(F.avg("len"))
                    .collect()[0][0]
                )
                if avgWordCount is not None and avgWordCount > 15:
                    tokenizer = Tokenizer(inputCol=colname, outputCol=colname + "_tokens")
                    hashingTf = HashingTF(inputCol=colname + "_tokens", outputCol=colname + "_tf")
                    stages.extend([tokenizer, hashingTf])
                    finalFeatures.append(colname + "_tf")
                else:
                    indexer = StringIndexer(inputCol=colname, outputCol=colname + "_idx", handleInvalid="skip")
                    stages.append(indexer)
                    finalFeatures.append(colname + "_idx")
            else:
                finalFeatures.append(colname)
        stages.append(VectorAssembler(inputCols=finalFeatures, outputCol="features"))
        labelCol = target
        if isinstance(sparkDf.schema[target].dataType, StringType):
            labelCol = target + "_idx"
            stages.append(StringIndexer(inputCol=target, outputCol=labelCol, handleInvalid="skip"))
        models = {
            "Logistic Regression": LogisticRegression(featuresCol="features", labelCol=labelCol, maxIter=10),
            "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol=labelCol),
            "Random Forest": RandomForestClassifier(featuresCol="features", labelCol=labelCol, numTrees=20)
        }
        results = []
        train, test = sparkDf.randomSplit([0.8, 0.2], seed=42)
        for name, model in models.items():
            pipeline = Pipeline(stages=stages + [model])
            try:
                fitted = pipeline.fit(train)
                preds = fitted.transform(test)
                evaluatorF1 = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="f1")
                evaluatorPrecision = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="weightedPrecision")
                evaluatorRecall = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="weightedRecall")
                evaluatorAcc = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="accuracy")
                f1 = evaluatorF1.evaluate(preds)
                precision = evaluatorPrecision.evaluate(preds)
                recall = evaluatorRecall.evaluate(preds)
                accuracy = evaluatorAcc.evaluate(preds)
                results.append((
                    name,
                    round(accuracy * 100, 2),
                    round(f1 * 100, 2),
                    round(precision * 100, 2),
                    round(recall * 100, 2)
                ))
            except Exception as e:
                results.append((name, f"Failed: {str(e).splitlines()[0]}"))
        return results

    # Train and evaluate a text classification model using TF-IDF and Naive Bayes
    def evaluateTextClassificationModels(self, sparkDf, target):
        featureCols = [col for col in sparkDf.columns if col != target]
        stages = []
        finalFeatures = []
        timestampCols = [f.name for f in sparkDf.schema.fields if isinstance(f.dataType, TimestampType)]
        for tsCol in timestampCols:
            sparkDf = sparkDf.withColumn(f"{tsCol}_year", F.year(F.col(tsCol)))
            sparkDf = sparkDf.withColumn(f"{tsCol}_month", F.month(F.col(tsCol)))
            sparkDf = sparkDf.withColumn(f"{tsCol}_day", F.dayofmonth(F.col(tsCol)))
            sparkDf = sparkDf.drop(tsCol)
        sparkDf = sparkDf.filter(F.col(target).isNotNull())
        stringCols = [f.name for f in sparkDf.schema.fields if isinstance(f.dataType, StringType)]
        numericCols = [f.name for f in sparkDf.schema.fields if isinstance(f.dataType, NumericType)]
        sparkDf = sparkDf.fillna("missing", subset=stringCols)
        sparkDf = sparkDf.fillna(0, subset=numericCols)
        for colname in featureCols:
            if colname not in sparkDf.columns:
                continue
            dtype = sparkDf.schema[colname].dataType
            if isinstance(dtype, StringType):
                tokenizer = Tokenizer(inputCol=colname, outputCol=colname + "_tokens")
                countVec = CountVectorizer(inputCol=colname + "_tokens", outputCol=colname + "_tf", vocabSize=100000)
                idf = IDF(inputCol=colname + "_tf", outputCol=colname + "_tfidf")
                stages.extend([tokenizer, countVec, idf])
                finalFeatures.append(colname + "_tfidf")
            else:
                finalFeatures.append(colname)
        stages.append(VectorAssembler(inputCols=finalFeatures, outputCol="features"))
        labelCol = target
        if isinstance(sparkDf.schema[target].dataType, StringType):
            labelCol = target + "_idx"
            stages.append(StringIndexer(inputCol=target, outputCol=labelCol, handleInvalid="skip"))
        naiveBayesModel = NaiveBayes(featuresCol="features", labelCol=labelCol)
        results = []
        train, test = sparkDf.randomSplit([0.8, 0.2], seed=42)
        pipeline = Pipeline(stages=stages + [naiveBayesModel])
        try:
            fitted = pipeline.fit(train)
            preds = fitted.transform(test)
            evaluatorF1 = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="f1")
            evaluatorPrecision = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="weightedPrecision")
            evaluatorRecall = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="weightedRecall")
            evaluatorAcc = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="accuracy")
            f1 = evaluatorF1.evaluate(preds)
            precision = evaluatorPrecision.evaluate(preds)
            recall = evaluatorRecall.evaluate(preds)
            accuracy = evaluatorAcc.evaluate(preds)
            results.append((
                "Naive Bayes",
                round(accuracy * 100, 2),
                round(f1 * 100, 2),
                round(precision * 100, 2),
                round(recall * 100, 2)
            ))
        except Exception as e:
            results.append(("Naive Bayes", f"Failed: {str(e).splitlines()[0]}"))
        return results
