import shutil
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, FloatType, LongType, StringType
from pyspark.ml.feature import StringIndexer

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from utils._helper import Helper

st.set_page_config(layout="wide")

# Spark initialization (runs once)
if "spark" not in st.session_state:
    st.session_state.spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("DataLoadComparison")
        .getOrCreate()
    )

# Initialize session state keys
defaultKeys = ["loadDataset", "pysparkDf", "pandasDf", "datasetName", "modelType", "workingDf", "datasetVersion"]
for key in defaultKeys:
    st.session_state.setdefault(key, None)

# Sidebar navigation
st.sidebar.title("Automated Data Cleaning Pipeline for ML")
section = st.sidebar.selectbox(
    "Choose a Section", [
        "Dataset Loading Time Comparison",
        "Issues in Dataset",
        "Data Profile Report",
        "Clean the Data",
        "Performance Analysis"
    ]
)

# Section 1: Dataset Loading Time Comparison
if section == "Dataset Loading Time Comparison":
    uploadedFile = st.file_uploader("Upload your CSV file", type=["csv"])
    modelType = st.selectbox("Select model type", ("Classification", "Regression", "Text Classification"))

    def loadDataset(uploadedFile, modelType):
        if not uploadedFile:
            st.error("Please upload a dataset first.")
            return
        try:
            if os.path.exists("UploadedDatasets"):
                shutil.rmtree("UploadedDatasets")

            helper = Helper(uploadedFile.name)
            st.text("Loading data with Pandas and PySpark... please wait...")
            pandasDf, pysparkDf, pandasTime, pysparkTime = helper.compareLoadingTimes(uploadedFile)

            st.session_state.update({
                "pandasDf": pandasDf,
                "pysparkDf": pysparkDf,
                "datasetName": uploadedFile.name,
                "workingDf": pysparkDf,
                "datasetVersion": uploadedFile.name,
                "modelType": modelType,
                "loadDataset": True
            })

            st.write(f"Time to load {uploadedFile.name} using Pandas: {pandasTime:.2f} seconds")
            st.write(f"Time to load {uploadedFile.name} using PySpark: {pysparkTime:.2f} seconds")
            st.write("Pandas is faster!" if pandasTime < pysparkTime else "PySpark is faster!")
            st.write("First 5 rows of the dataset:")
            st.write(pandasDf.head(5))

        except Exception as e:
            st.error(f"An error occurred while loading the dataset: {e}")

    if uploadedFile and st.button("Load Dataset"):
        loadDataset(uploadedFile, modelType)

# Section 2: Issues in Dataset
elif section == "Issues in Dataset":
    st.header("Issues in Dataset")

    if st.session_state.loadDataset and st.session_state.pysparkDf is not None:
        predictingColumn = st.selectbox(
            "Select the column you're trying to predict:",
            st.session_state.pysparkDf.columns
        )
        st.session_state["predictingColumn"] = predictingColumn

        if st.button("Load Issues"):
            helper = Helper(st.session_state.datasetName)

            # 0. Dataset Schema
            schemaInfo = [
                (field.name, field.dataType.simpleString())
                for field in st.session_state.pysparkDf.schema.fields
            ]
            schemaDf = pd.DataFrame(schemaInfo, columns=["Column", "Data Type"])
            st.header("Dataset Schema")
            st.dataframe(schemaDf)

            # 1. Missing Values
            st.header("1. Missing Values Detection")
            st.session_state.pysparkDf = helper.extractTimestampFeatures(st.session_state.pysparkDf)
            missingSummaryDf, rowsWithMissing = helper.detectMissingValues(st.session_state.pysparkDf)
            st.subheader("Missing Value Count Per Column")

            colsWithMissing = (
                missingSummaryDf[missingSummaryDf["Missing Count"] > 0]["Column"].tolist()
            )
            if colsWithMissing:
                st.subheader("Columns with Missing Values")
                st.write(", ".join(colsWithMissing))

            missingPie = missingSummaryDf[missingSummaryDf["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(missingSummaryDf)

            # 2. Duplicate Rows
            st.header("2. Duplicate Rows Detection")
            columnsToCheck = [col for col in st.session_state.pysparkDf.columns if col != predictingColumn]
            duplicateColumns, duplicateRowsPd = helper.detectDuplicateRows(st.session_state.pysparkDf, columnsToCheck)
            st.subheader("Columns Checked for Duplicates")
            st.write(", ".join(duplicateColumns))

            duplicateCount = duplicateRowsPd.shape[0]
            col3, col4 = st.columns([3, 1])
            with col3:
                st.write(f"Duplicate Rows: `{duplicateCount}`")

            # 3. Skewness
            st.header("3. Skewness Detection")
            skewnessDf = helper.detectSkewness(st.session_state.pysparkDf, predictingColumn)
            if not skewnessDf.empty:
                st.subheader("Skewness of Columns (excluding predicting column)")
                st.dataframe(skewnessDf)
            else:
                st.info("No numeric columns available for skewness analysis.")

            # 4. Outlier Detection
            st.header("4. Outlier Detection")
            outlierDf = helper.detectOutliers(st.session_state.pysparkDf, predictingColumn)
            if not outlierDf.empty:
                st.subheader("Outliers per Column (IQR Method)")
                st.dataframe(outlierDf)

                numericCols = [
                    f.name for f in st.session_state.pysparkDf.schema.fields
                    if isinstance(f.dataType, (DoubleType, IntegerType, FloatType, LongType))
                ]
                if numericCols:
                    outlierData = st.session_state.pysparkDf.select(numericCols).toPandas()
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    sns.boxplot(data=outlierData, ax=ax3)
                    ax3.set_title("Boxplot for Outlier Detection")
                    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
                    st.pyplot(fig3)
                else:
                    st.info("No numeric columns found.")
            else:
                st.info("No numeric columns or no outliers detected.")

            # 5. Correlation
            st.header("5. Correlation with Predicting Column")
            correlationDf = helper.detectCorrelation(st.session_state.pysparkDf, predictingColumn)
            if not correlationDf.empty:
                st.subheader(f"Correlation between Features and Target: '{predictingColumn}'")
                st.dataframe(correlationDf)
            else:
                st.info("No valid numeric or low-cardinality categorical features found for correlation.")

            # 6. Class Imbalance
            if st.session_state.modelType.lower() in ["classification", "text classification"]:
                st.header("6. Class Imbalance Detection")
                classDist = st.session_state.pysparkDf.groupBy(st.session_state.predictingColumn).count().toPandas()
                if not classDist.empty:
                    st.write("Class Distribution:")
                    st.dataframe(classDist)

                    fig4, ax4 = plt.subplots(figsize=(8, 6))
                    sns.barplot(x=predictingColumn, y="count", data=classDist, ax=ax4, palette="Blues_d")
                    ax4.set_title("Class Imbalance")
                    ax4.set_xlabel("Class Labels")
                    ax4.set_ylabel("Count")
                    for p in ax4.patches:
                        ax4.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                                     textcoords='offset points')
                    st.pyplot(fig4)
                else:
                    st.info("Could not compute class distribution.")
    else:
        st.warning("Please load a dataset first from the 'Dataset Loading Time Comparison' section.")
            
# Section 3: Data Profile Report
elif section == "Data Profile Report":
    st.header("Data Profile Report")

    if st.session_state.loadDataset and st.session_state.pandasDf is not None:
        profile = ProfileReport(
            st.session_state.pandasDf,
            title="Data Profile (Minimal)",
            explorative=True,
            minimal=True
        )
        st_profile_report(profile)
    else:
        st.warning("Please load a dataset first from the 'Dataset Loading Time Comparison' section.")
        
# Section 4: Clean the Data
elif section == "Clean the Data":
    st.header("Clean the Data")

    if st.session_state.loadDataset and st.session_state.pysparkDf is not None:
        helper = Helper(st.session_state.datasetName)

        if "workingDf" not in st.session_state or st.session_state.get("datasetVersion") != st.session_state.datasetName:
            st.session_state.workingDf = st.session_state.pysparkDf
            st.session_state.datasetVersion = st.session_state.datasetName

        workingDf = st.session_state.workingDf

        st.subheader("Select Predicting Column")
        predictingColumn = st.selectbox(
            "Which column are you trying to predict?",
            workingDf.columns,
            index=None,
            placeholder="Select a column"
        )

        if predictingColumn:
            st.session_state["predictingColumn"] = predictingColumn
            if isinstance(workingDf.schema[predictingColumn].dataType, StringType):
                labelIndexer = StringIndexer(inputCol=predictingColumn, outputCol="__indexed_label", handleInvalid="skip")
                model = labelIndexer.fit(workingDf)
                workingDf = model.transform(workingDf).drop(predictingColumn).withColumnRenamed("__indexed_label", predictingColumn)
                st.session_state.workingDf = workingDf
                st.success(f"Converted target column '{predictingColumn}' to numeric labels.")
            print("Columns after dropping duplicates:", workingDf.columns)

        st.subheader("1. Handle Missing Values")
        missingSummaryDf, _ = helper.detectMissingValues(workingDf)
        missingCols = missingSummaryDf[missingSummaryDf["Missing Count"] > 0]["Column"].tolist()

        numericMissing = [
            field.name for field in workingDf.schema.fields
            if field.name in missingCols and isinstance(field.dataType, (IntegerType, FloatType, DoubleType, LongType))
        ]

        if numericMissing:
            missingStrategy = st.selectbox(
                "Choose missing value strategy:",
                ["Do Nothing", "Drop Rows", "Fill with Mean (numeric only)", "Fill with Median (numeric only)", "Fill with Zero", "Fill with 'Unknown'"]
            )
        else:
            missingStrategy = st.selectbox(
                "Choose missing value strategy:",
                ["Do Nothing", "Drop Rows"]
            )

        if st.button("Apply Missing Value Strategy"):
            if missingStrategy != "Do Nothing":
                workingDf = helper.handleMissingValuesSpark(workingDf, missingStrategy)
                st.session_state.workingDf = workingDf
                st.success(f"Missing values handled using: {missingStrategy}")

        temp = st.session_state.predictingColumn if "predictingColumn" in st.session_state else None

        st.subheader("2. Remove Duplicate Rows")
        if st.button("Drop Duplicates"):
            before = workingDf.count()
            featureCols = [col for col in workingDf.columns if col != temp]
            print("Columns before dropping duplicates:", workingDf.columns)
            workingDf = helper.removeDuplicatesSpark(workingDf.select(*featureCols, temp))
            print("Columns after dropping duplicates:", workingDf.columns)
            after = workingDf.count()
            st.session_state.workingDf = workingDf
            st.success(f"Removed {before - after} duplicate rows.")

        st.subheader("3. Modify Outliers")
        if st.button("Remove Numeric Outliers (5xIQR)"):
            before = workingDf.count()
            workingDf, changedRows = helper.removeNumericOutliers(workingDf, temp)
            after = workingDf.count()
            st.session_state.workingDf = workingDf
            st.success(f"Removed {changedRows} rows due to numeric outliers.")

        if st.session_state.modelType.lower() == "regression":
            st.subheader("Fix Very High Skew (Regression Only)")
            skewMethod = st.selectbox(
                "Choose transformation method:",
                ["log", "sqrt", "cbrt"],
                key="regSkewMethod"
            )

            if st.button("Apply Skewness Transformation (|skew|>1)"):
                skewed = helper.detectHighSkewNumeric(workingDf, threshold=1.0)
                if skewed:
                    colsToFix = [col for col, sk in skewed]
                    st.write("Columns with |skew| > 1:", ", ".join(f"{c} (skew={s})" for c, s in skewed))
                    workingDf, fixedCols = helper.fixHighSkewForRegression(workingDf, method=skewMethod, threshold=1.0)
                    st.session_state.workingDf = workingDf
                    if fixedCols:
                        st.success(f"Applied {skewMethod} to: {', '.join(fixedCols)}")
                else:
                    st.info("No numeric columns (including target) had |skew| > 1.")

        st.subheader("4. Basic Text Cleaning")
        textFields = [f.name for f in workingDf.schema.fields if isinstance(f.dataType, StringType) and f.name != predictingColumn]

        if textFields:
            selectedCols = st.multiselect("Select columns to clean:", textFields, key="textCleanCols")

            if st.button("Perform Text Cleaning"):
                if selectedCols:
                    workingDf = helper.cleanTextColumns(workingDf, selectedCols)
                    st.session_state.workingDf = workingDf
                    st.success("Text cleaning applied to selected columns.")
                else:
                    st.warning("Please select at least one column.")
        else:
            st.info("No text columns available for cleaning.")

        if st.session_state.modelType.lower() in ["classification", "text classification"]:
            st.subheader("5. Resolve Class Imbalance")
            classDist = workingDf.groupBy(temp).count().toPandas()
            st.write("🔹 Class Distribution (Before Resampling)")
            st.dataframe(classDist)

            minClass = classDist["count"].min()
            maxClass = classDist["count"].max()

            if maxClass > 1.5 * minClass:
                st.warning("Significant class imbalance detected.")
                strategy = st.radio(
                    "Choose Resampling Strategy:",
                    options=["Random OverSampler", "Random UnderSampler"]
                )

                if st.button("Apply Resampling Strategy"):
                    if strategy == "Random OverSampler":
                        workingDf = helper.randomOversample(workingDf, temp)
                        st.success("Applied Random OverSampling.")
                    else:
                        workingDf = helper.undersample(workingDf, temp)
                        st.success("Applied Random UnderSampling.")

                    st.session_state.workingDf = workingDf

                    newDist = workingDf.groupBy(temp).count().toPandas()
                    st.write("Class Distribution (After Resampling)")
                    st.dataframe(newDist)

                    fig, ax = plt.subplots()
                    sns.barplot(x=temp, y="count", data=newDist, ax=ax, palette="Blues_d")
                    ax.set_title("Class Distribution After Resampling")
                    st.pyplot(fig)
            else:
                st.info("No significant class imbalance detected.")

        st.subheader("Final Dataset Summary")
        if st.button("Show Final Cleaned Dataset"):
            st.write("Before Cleaning:", st.session_state.pysparkDf.count(), "rows")
            st.write("After Cleaning:", st.session_state.workingDf.count(), "rows")

            st.write("Before Cleaning dataset:")
            st.dataframe(st.session_state.pysparkDf.limit(50).toPandas(), use_container_width=True)

            st.write("After Cleaning dataset (first 50 rows):")
            cleanedPreviewDf = st.session_state.workingDf.limit(50).toPandas()
            st.dataframe(cleanedPreviewDf, use_container_width=True)
            st.session_state.cleanedDfPreview = cleanedPreviewDf

            if st.button("Generate Full Cleaned CSV"):
                fullCleanedDf = st.session_state.workingDf.toPandas()
                st.session_state.cleanedCsv = fullCleanedDf.to_csv(index=False)
                st.success("Cleaned CSV is ready for download!")

        if "cleanedCsv" in st.session_state:
            st.download_button(
                label="Download Full Cleaned Dataset",
                data=st.session_state.cleanedCsv,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

        if st.button("Reset Cleaning"):
            st.session_state.workingDf = st.session_state.pysparkDf
            st.success("Reset to original dataset.")

    else:
        st.warning("Please load a dataset first from the 'Dataset Loading Time Comparison' section.")

# Section 5: Performance Analysis
elif section == "Performance Analysis":
    st.header("Performance Analysis")

    if not st.session_state.loadDataset:
        st.warning("Please load a dataset first.")
    else:
        rawDf = st.session_state.pysparkDf
        cleanedDf = st.session_state.workingDf if "workingDf" in st.session_state else None

        target = st.selectbox("Select Target (Predicting) Column", rawDf.columns)
        helper = Helper(st.session_state.datasetName)

        if st.button("Run Analysis"):
            modelType = st.session_state.modelType.lower()

            if modelType == "regression":
                st.header("Performance Analysis: Regression Models")

                st.subheader("On Raw Dataset")
                rawResults = helper.evaluateRegressionModels(rawDf, target)
                st.table(pd.DataFrame(rawResults, columns=["Model", "RMSE", "MSE"]))

                if cleanedDf is not None:
                    st.subheader("On Cleaned Dataset")
                    cleanedResults = helper.evaluateRegressionModels(cleanedDf, target)
                    st.table(pd.DataFrame(cleanedResults, columns=["Model", "RMSE", "MSE"]))
                else:
                    st.info("No cleaned dataset found. Clean your data first.")

            elif modelType == "classification":
                st.header("Performance Analysis: Classification Models")

                st.subheader("On Raw Dataset")
                rawDf = helper.extractTimestampFeatures(rawDf)
                rawResults = helper.evaluateClassificationModels(rawDf, target)
                st.table(pd.DataFrame(rawResults, columns=["Model", "Accuracy", "F1 Score", "Precision", "Recall"]))

                if cleanedDf is not None:
                    st.subheader("On Cleaned Dataset")
                    cleanedDf = helper.extractTimestampFeatures(cleanedDf)
                    cleanedResults = helper.evaluateClassificationModels(cleanedDf, target)
                    st.table(pd.DataFrame(cleanedResults, columns=["Model", "Accuracy", "F1 Score", "Precision", "Recall"]))
                else:
                    st.info("No cleaned dataset found. Clean your data first.")

            elif modelType == "text classification":
                st.header("Performance Analysis: Text Classification Models")

                st.subheader("On Raw Dataset")
                rawResults = helper.evaluateTextClassificationModels(rawDf, target)
                st.table(pd.DataFrame(rawResults, columns=["Model", "Accuracy", "F1 Score", "Precision", "Recall"]))

                if cleanedDf is not None:
                    st.subheader("On Cleaned Dataset")
                    cleanedResults = helper.evaluateTextClassificationModels(cleanedDf, target)
                    st.table(pd.DataFrame(cleanedResults, columns=["Model", "Accuracy", "F1 Score", "Precision", "Recall"]))
                else:
                    st.info("No cleaned dataset found. Clean your data first.")