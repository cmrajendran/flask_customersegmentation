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

#segmentation
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

            # Map segment labels
            segment_labels = {
                0: "Average Customers",
                1: "High-Spending Infrequent Customers",
                2: "Inactive Customers",
                3: "Loyal and Active Customers"
            }
            df_rfm_cleaned["segment_label"] = df_rfm_cleaned["segments"].map(segment_labels)

            # Perform cluster analysis
            numeric_columns = df_rfm_cleaned.select_dtypes(include=["number"]).columns
            cluster_analysis = (
                df_rfm_cleaned.groupby("segment_label")[numeric_columns]
                .mean()
                .to_html(classes="table table-striped")
            )

            # Define actions table for segments
            actions = [
                {"Segment": "Loyal and Active Customers", "Action": "Implement loyalty programs, exclusive previews, and suggest complementary products."},
                {"Segment": "High-Spending Infrequent Customers", "Action": "Send reminders, offer subscription models, and run retargeting campaigns."},
                {"Segment": "Average Customers", "Action": "Highlight discounts, bundles, and create urgency with flash sales."},
                {"Segment": "Inactive Customers", "Action": "Win-back campaigns with discounts and analyze reasons for inactivity."}
            ]
            actions_table = pd.DataFrame(actions).to_html(classes="table table-striped", index=False)

            # Generate pie chart for segment percentages
            segment_counts = df_rfm_cleaned["segments"].value_counts()
            segment_percentages = (segment_counts / segment_counts.sum()) * 100

            legend_labels = [f"Cluster {cluster}: {label}" for cluster, label in segment_labels.items()]

            plt.figure(figsize=(8, 8))
            plt.pie(
                segment_percentages,
                labels=segment_counts.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=sns.color_palette("viridis", len(segment_counts)),
            )
            plt.title("Clustered Segments by Percentage", fontsize=16)
            # Add legend
            plt.legend(legend_labels, title="Segments", loc="center left", bbox_to_anchor=(1, 0.5))
            plt.tight_layout()

            # Save pie chart
            pie_chart_path = os.path.join(app.config["UPLOAD_FOLDER"], "cluster_pie_chart.png")
            plt.savefig(pie_chart_path)
            plt.close()

            # Select 20 random customers and prepare for download
            sample_customers = df_rfm_cleaned.sample(n=20, random_state=42)[["customer_id", "segments", "segment_label"]]
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
                actions_table=actions_table
            )
        except Exception as e:
            return render_template("segmentation.html", error=f"An error occurred: {str(e)}")

    # Handle the GET request
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



#RFM
@app.route("/rfm", methods=["GET", "POST"])
def rfm():
    if request.method == "POST":
        # Retrieve the uploaded CSV files
        customers_file = request.files.get("customers_file")
        payments_file = request.files.get("payments_file")
        orders_file = request.files.get("orders_file")

        if not customers_file or not payments_file or not orders_file:
            return render_template("rfm.html", error="Please upload all required CSV files: customers, payments, and orders.")

        try:
            # Load datasets
            customers = pd.read_csv(customers_file)
            payments = pd.read_csv(payments_file)
            orders = pd.read_csv(orders_file)

            # Convert date columns into datetime format
            date_cols = [
                'order_purchase_timestamp',
                'order_approved_at',
                'order_delivered_carrier_date',
                'order_delivered_customer_date',
                'order_estimated_delivery_date'
            ]
            for col in date_cols:
                if col in orders.columns:
                    orders[col] = pd.to_datetime(orders[col], errors="coerce")

            # Merge datasets
            order_customers = orders.merge(customers, on="customer_id", how="inner")
            df_complete = order_customers.merge(payments, on="order_id", how="inner")

            # Filter for delivered orders
            df_complete = df_complete[df_complete["order_status"] == "delivered"]

            # Drop unnecessary columns
            columns_to_drop = ["order_status", "order_purchase_timestamp"]
            df_full = df_complete.drop(columns=[col for col in columns_to_drop if col in df_complete.columns])

            # Calculate Recency
            max_date = df_full["order_approved_at"].max()
            recency_df = df_full.groupby("customer_id")["order_approved_at"].max().reset_index()
            recency_df["recency"] = (max_date - recency_df["order_approved_at"]).dt.days

            # Calculate Frequency
            frequency_df = df_full.groupby("customer_id")["order_id"].nunique().reset_index()
            frequency_df = frequency_df.rename(columns={"order_id": "frequency"})

            # Calculate Monetary Value
            monetary_df = df_full.groupby("customer_id")["payment_value"].sum().reset_index()
            monetary_df = monetary_df.rename(columns={"payment_value": "monetary"})

            # Merge Recency, Frequency, and Monetary dataframes
            rfm_df = recency_df.merge(frequency_df, on="customer_id").merge(monetary_df, on="customer_id")

            # Drop the `order_approved_at` column and missing values
            rfm_cleaned = rfm_df.drop(columns=["order_approved_at"]).dropna()

            # Save the cleaned RFM dataset
            rfm_cleaned_path = os.path.join(app.config["UPLOAD_FOLDER"], "rfm_cleaned.csv")
            rfm_cleaned.to_csv(rfm_cleaned_path, index=False)

            # Display the current RFM preview with `customer_id`
            rfm_preview = rfm_cleaned.head(5)

            # Drop the `customer_id` column for the new preview
            rfm_no_id = rfm_cleaned.drop(columns=["customer_id"]).head(5)

            # Render the results
            return render_template(
                "rfm.html",
                message="RFM dataset successfully created and saved.",
                rfm_data=rfm_preview.to_html(classes="table table-striped"),
                rfm_no_id_data=rfm_no_id.to_html(classes="table table-striped"),
                download_csv_path="static/rfm_cleaned.csv"
            )

        except Exception as e:
            # Handle exceptions and return meaningful error messages
            return render_template("rfm.html", error=f"An error occurred: {str(e)}")

    return render_template("rfm.html")




if __name__ == "__main__":
    app.run(debug=True)