import pandas as pd
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the input CSV file")
    parser.add_argument("--output_path", help="Path to the output CSV file")
    parser.add_argument("--limit_number", help="Number of rows to limit the output to", type=int, default=None)
    args = parser.parse_args()
    df = pd.read_csv(args.input_path)
    df = df.join(df['.geo'].apply(json.loads).apply(pd.Series))
    df[['longitude', 'latitude']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)
    print(df['latitude'].min())
    df = df[["latitude", "longitude", "year", "class"]]
    df['observation_time'] = pd.to_datetime(df['year'], format='%Y')
    df['sample_index'] = df.index
    if args.limit_number is not None:
        df = df.head(args.limit_number)
    df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()