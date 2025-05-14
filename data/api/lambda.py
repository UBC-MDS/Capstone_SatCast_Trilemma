import boto3
import io
import requests
import pandas as pd
from datetime import datetime

S3_BUCKET = "ENTER_BUCKET_NAME"

def fetch_data():
    timestamp = int(datetime.utcnow().timestamp())
    data = {"timestamp": timestamp}
    
    r = requests.get("https://mempool.space/api/v1/fees/mempool-blocks")
    if r.ok:
        block_data = r.json()[0]
        block_data.pop("feeRange", None)
        data.update({f"mempool_blocks_{k}": v for k, v in block_data.items()})
    
    r = requests.get("https://mempool.space/api/v1/fees/recommended")
    if r.ok:
        data.update({f"recommended_fee_{k}": v for k, v in r.json().items()})
    
    r = requests.get("https://mempool.space/api/mempool")
    if r.ok:
        data.update({f"mempool_{k}": v for k, v in r.json().items()})
    
    r = requests.get("https://mempool.space/api/v1/difficulty-adjustment")
    if r.ok:
        data.update({f"difficulty_adjustment_{k}": v for k, v in r.json().items()})
    
    r = requests.get("https://mempool.space/api/v1/prices")
    if r.ok:
        data.update({f"price_{k}": v for k, v in r.json().items()})
    
    return flatten_data(data)

def flatten_data(data):
    fee_hist_key = "mempool_fee_histogram"
    if fee_hist_key in data and isinstance(data[fee_hist_key], list):
        histogram = data.pop(fee_hist_key)
        bins = []
        for lower in range(1, 10):
            bins.append((lower, lower+1, f"{lower}_{lower+1}"))
        for lower in range(10, 30, 2):
            bins.append((lower, lower+2, f"{lower}_{lower+2}"))
        for lower in range(30, 100, 5):
            bins.append((lower, lower+5, f"{lower}_{lower+5}"))
        for lower in range(100, 1000, 50):
            bins.append((lower, lower+50, f"{lower}_{lower+50}"))
        bins.append((1000, float('inf'), "1000_plus"))
        
        bin_counts = {f"mempool_fee_histogram_bin_{label}": 0 for _, _, label in bins}
        for pair in histogram:
            if isinstance(pair, list) and len(pair) == 2:
                fee, count = pair
                for lower, upper, label in bins:
                    if lower <= fee < upper:
                        bin_counts[f"mempool_fee_histogram_bin_{label}"] += count
                        break
        data.update(bin_counts)
    
    return data

def lambda_handler(event, context):
    record = fetch_data()
    df = pd.DataFrame([record]).astype("float64")
    # Drop date columns
    drop_cols = [
        "difficulty_adjustment_estimatedRetargetDate",
        "difficulty_adjustment_previousRetarget",
        "price_time"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    
    # Write the DataFrame to a Parquet file in memory
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    
    timestamp = record["timestamp"]
    key = f"mempool_data_{timestamp}.parquet"
    
    # Upload to S3
    s3 = boto3.client("s3")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buffer.getvalue())
    
    return {
        "statusCode": 200,
        "body": f"Data saved to s3://{S3_BUCKET}/{key}"
    }
