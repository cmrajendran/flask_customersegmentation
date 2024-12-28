import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



app = Flask(__name__)

# Directory to save static files
UPLOAD_FOLDER = os.path.join("static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure static directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Home Route
@app.route("/")
def home():
    return render_template("home.html")

#Segmentation
@app.route("/segmentation", methods=["GET", "POST"])
def segmentation():
    if request.method == "POST":
        try:
            # Process uploaded files and perform segmentation
            rfm_file = request.files.get("rfm_file")
            kmeans_file = request.files.get("kmeans_file")

            if not rfm_file or not kmeans_file:
                return render_template("segmentation.html", error="Both files are required.")

            # Load RFM dataset and KMeans model
            df_rfm_cleaned = pd.read_csv(rfm_file)
            kmeans = pickle.load(kmeans_file)

            # Standardize features
            features = df_rfm_cleaned.drop(columns=["customer_id"])
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Predict clusters
            cluster_labels = kmeans.predict(features_scaled)
            df_rfm_cleaned["segments"] = cluster_labels

            # Perform cluster analysis
            numeric_columns = df_rfm_cleaned.select_dtypes(include=["number"]).columns
            cluster_analysis = (
                df_rfm_cleaned.groupby("segments")[numeric_columns]
                .mean()
                .to_html(classes="table table-striped")
            )

            # Generate pie chart for segment percentages
            segment_counts = df_rfm_cleaned["segments"].value_counts()
            segment_percentages = (segment_counts / segment_counts.sum()) * 100

            plt.figure(figsize=(8, 8))
            plt.pie(
                segment_percentages,
                labels=segment_counts.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=sns.color_palette("viridis", len(segment_counts)),
            )
            plt.title("Clustered Segments by Percentage", fontsize=16)
            plt.tight_layout()

            # Save pie chart
            pie_chart_path = os.path.join(app.config["UPLOAD_FOLDER"], "cluster_pie_chart.png")
            plt.savefig(pie_chart_path)
            plt.close()

            # Select 20 random customers and prepare for download
            sample_customers = df_rfm_cleaned.sample(n=20, random_state=42)[["customer_id", "segments"]]
            download_csv_path = os.path.join(app.config["UPLOAD_FOLDER"], "sample_customers.csv")
            sample_customers.to_csv(download_csv_path, index=False)

            # Select 5 customers for display
            display_customers = sample_customers.head(5)

            # Render results
            return render_template(
                "segmentation.html",
                cluster_analysis=cluster_analysis,
                processed_data=display_customers.to_html(classes="table table-striped", index=False),
                download_csv_path="static/sample_customers.csv",
                pie_chart="static/cluster_pie_chart.png",
            )
        except Exception as e:
            return render_template("segmentation.html", error=f"An error occurred: {str(e)}")

    return render_template("segmentation.html")






# EDA Section
@app.route("/eda", methods=["GET", "POST"])
def eda():
    if request.method == "POST":
        try:
            # Get uploaded files
            uploaded_files = request.files.getlist("datasets")
            if not uploaded_files:
                return render_template("eda.html", error="No files uploaded. Please upload CSV files.")

            datasets = {}
            plots = []

            # Process each uploaded dataset
            for uploaded_file in uploaded_files:
                # Validate file format
                if not uploaded_file.filename.endswith(".csv"):
                    return render_template("eda.html", error=f"{uploaded_file.filename} is not a CSV file.")

                dataset_name = uploaded_file.filename
                dataset = pd.read_csv(uploaded_file)

                # Summarize dataset
                datasets[dataset_name] = {
                    "info": {
                        "rows": dataset.shape[0],
                        "columns": dataset.shape[1],
                        "columns_list": dataset.columns.tolist(),
                    },
                    "head": dataset.head(5).to_html(classes="table table-striped"),
                    "stats": dataset.describe().to_html(classes="table table-striped"),
                }

                # Generate visualizations
                numeric_columns = dataset.select_dtypes(include=["number"]).columns
                if len(numeric_columns) > 0:
                    # Histogram of the first numeric column
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data=dataset, x=numeric_columns[0], kde=True)
                    plt.title(f"Distribution of {numeric_columns[0]} in {dataset_name}", fontsize=16)
                    hist_path = os.path.join(UPLOAD_FOLDER, f"{dataset_name}_hist.png")
                    plt.savefig(hist_path)
                    plt.close()
                    plots.append(hist_path)

                    # Heatmap for numeric columns
                    if len(numeric_columns) > 1:
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(dataset[numeric_columns].corr(), annot=True, cmap="coolwarm")
                        heatmap_path = os.path.join(UPLOAD_FOLDER, f"{dataset_name}_heatmap.png")
                        plt.savefig(heatmap_path)
                        plt.close()
                        plots.append(heatmap_path)

            # Return results to template
            return render_template("eda.html", datasets=datasets, plots=plots)

        except Exception as e:
            return render_template("eda.html", error=f"An error occurred: {str(e)}")

    return render_template("eda.html")


if __name__ == "__main__":
    app.run(debug=True)