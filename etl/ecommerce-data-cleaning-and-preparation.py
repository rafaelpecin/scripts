import pandas as pd

def main():
    # 1. Load the CSV file
    df = pd.read_csv("ecommerce_data.csv")

    # 2. Display the first 5 rows
    print("=== First 5 Rows ===")
    print(df.head(), "\n")

    # 3. Convert conversion_status to numeric (1 = Converted, 0 = Did Not Convert)
    df["conversion_status_numeric"] = df["conversion_status"].map({
        "Converted": 1,
        "Did Not Convert": 0
    })

    # Save the transformed dataset to a new file
    output_file = "ecommerce_data_processed.csv"
    df.to_csv(output_file, index=False)
    print(f"=== Processed data saved to: {output_file} ===\n")

    # 4. Correlation matrix for numeric columns
    print("=== Correlation Matrix ===")
    print(df.corr(numeric_only=True), "\n")

    # 5. Descriptive statistics grouped by device_type
    print("=== Descriptive Statistics by Device Type ===")
    grouped_stats = df.groupby("device_type")[["time_on_site_sec", "page_views"]].describe()
    print(grouped_stats)

if __name__ == "__main__":
    main()

