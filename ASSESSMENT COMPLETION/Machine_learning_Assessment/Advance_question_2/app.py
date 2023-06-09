from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import monotonically_increasing_id, col
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

from pyspark.ml.evaluation import BinaryClassificationEvaluator

import streamlit as st
import cloudpickle as pickle


spark = SparkSession.builder.appName("chemicals").getOrCreate()
filePath = 'indian_liver_patient.csv'
data=spark.read.format("csv").option("header","true").load(filePath)


# Define a function to preprocess the input data
def preprocess_input(data):
    # StringIndexer to convert the 'Age' column to numeric indices
    indexer = StringIndexer(inputCol='Gender', outputCol='GenderIndex')

    # OneHotEncoder to perform one-hot encoding on the 'AgeIndex' column
    encoder = OneHotEncoder(inputCols=['GenderIndex'], outputCols=['GenderVec'])

    # Create a pipeline to execute the StringIndexer and OneHotEncoder in sequence
    pipeline = Pipeline(stages=[indexer, encoder])
    encoded_df = pipeline.fit(data).transform(data)

    data = spark.createDataFrame(data)
    encoded_df = pipeline.fit(data).transform(data)
    df_with_index = encoded_df.withColumn("Index", monotonically_increasing_id())

    numeric_columns = df_with_index.select([col(column).cast("float").alias(column) for column in df_with_index.columns[2:] 
                                        if column not in ["GenderIndex", "GenderVec"]  
                                        ])
    encoded_column = df_with_index.select(col("GenderVec"),col("index"))
    # Concatenate the numeric_columns and encoded_columns
    all_columns = numeric_columns .join(encoded_column,on=["index"])
    all_columns = all_columns.drop("index")

    return all_columns

def train_model(data):
    inputCols = data.columns
    assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
    data = assembler.transform(data)
    train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)
    lr = LogisticRegression(labelCol="Dataset", featuresCol="features")
    model = lr.fit(train_data)
    model.save("model.pkl")

def evaluate_model(y_pred):
    evaluator = BinaryClassificationEvaluator(labelCol='Dataset', rawPredictionCol='prediction')
    classification_score = evaluator.evaluate(y_pred)
    print("Classification Score:", classification_score)

def make_predictions(data):
    # Preprocess the input data
    preprocessed_data = preprocess_input(data)
    # Convert the preprocessed data to a DMatrix
    model = LogisticRegressionModel.load("model.pkl")
    dmatrix = model.DMatrix(preprocessed_data)

    # Make predictions using the XGBoost model
    predictions = model.predict(dmatrix)

    return predictions
def main():
    # Set the app title
    st.title("Indian Liver Patient")

    # Create input fields for the features
    age = st.number_input("Age", value=12)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin", value=0.8)
    direct_bilirubin = st.number_input("Direct Bilirubin", value=0.2)
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase", value=302.0)
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase", value=47.0)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", value=67.0)
    total_protiens = st.number_input("Total Protiens", value=6.7)
    albumin = st.number_input("Albumin", value=3.5)
    albumin_globulin_ratio = st.number_input("Albumin and Globulin Ratio", value=1.1)

    # Create a dictionary with the input data
    input_data = {
        "Age": age,
        "Gender": gender,
        "Total_Bilirubin": total_bilirubin,
        "Direct_Bilirubin": direct_bilirubin,
        "Alkaline_Phosphotase": alkaline_phosphotase,
        "Alamine_Aminotransferase": alamine_aminotransferase,
        "Aspartate_Aminotransferase": aspartate_aminotransferase,
        "Total_Protiens": total_protiens,
        "Albumin": albumin,
        "Albumin_and_Globulin_Ratio": albumin_globulin_ratio
    }

    # Make predictions using the input data
    predictions = make_predictions(input_data)

    # Display the predictions
    st.header("Predictions")
    st.write(predictions)

if __name__ == "__main__":
    main()