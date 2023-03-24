import pandas as pd

if __name__ == "__main__":
    file_name = "sub_NS.csv"
    df = pd.read_csv(file_name, sep=";")
    result_df = None
    for i in range(2500):
        df_row = df.iloc[i, :]
        df_row_list = df_row[0].split(",")
        label = 1 if max(float(df_row_list[1]), float(df_row_list[2])) == float(df_row_list[2]) else 0
        slice_df = pd.DataFrame({
            "index": [i],
            "prediction": label
        })
        if result_df is None: 
            result_df = slice_df
        else:
            result_df = pd.concat([result_df,slice_df], axis=0)
    result_path = "NLPCC14-SC.tsv"
    result_df.to_csv(result_path, sep="\t", index=False)