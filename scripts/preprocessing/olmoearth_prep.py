import pandas as pd
import argparse
import geopandas as gpd
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to the input CSV file")
    parser.add_argument("--output_path", help="Path to the output CSV file")
    parser.add_argument("--limit_number", help="Number of rows to limit the output to", type=int, default=None)
    parser.add_argument("--split", help="Split label", type=str, default=None)
    args = parser.parse_args()
    df = pd.read_csv(args.input_path)
    df = df.join(df['.geo'].apply(json.loads).apply(pd.Series))
    df[['longitude', 'latitude']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)
    df = df[["latitude", "longitude", "year", "class"]]
    df['input_year'] = df['year'] - 1
    df['observation_time'] = pd.to_datetime(df['input_year'], format='%Y')
    df['sample_index'] = df.index
    df['split'] = args.split
    if args.limit_number is not None:
        df = df.head(args.limit_number)
    df.to_csv(args.output_path, index=False)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs='EPSG:4326'
    )
    gdf.to_file(args.output_path.replace('.csv', '.geojson'),
                driver='GeoJSON')

if __name__ == "__main__":
    main()
