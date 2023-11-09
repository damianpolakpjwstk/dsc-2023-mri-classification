from pathlib import Path

import pandas as pd


def parse_aseg_dkt_stats(path) -> pd.DataFrame:
    """
    Parse the aseg+DKT.stats file from FreeSurfer output.
    :param path: path to aseg+DKT.stats file.
    :return: DataFrame with parsed data.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if not line.startswith("#") or line.startswith("# ColHeaders")]
    lines_split = [[word for word in line.split(" ") if len(word) > 0 and word != "#"] for line in lines]
    colnames = lines_split[0]
    colnames.remove("ColHeaders")
    df = pd.DataFrame(lines_split[1:], columns=colnames)
    df = df[["Volume_mm3", "StructName", "normMean", "normStdDev", "normMin", "normMax"]]
    return df


def parse_directories(dir_path: str | Path) -> pd.DataFrame:
    """
    Parse all directories in the dir_path. Find the aseg+DKT.stats files and parse them.
    :param dir_path: Directory with FreeSurfer outputs.
    :return: Dataframe with parsed data. Each row is a single scan. Columns are the volumes of the structures.
    """
    loaded_scans_data = {}
    dir_path = Path(dir_path)
    aseg_dtk_paths = list(dir_path.glob("**/stats/aseg+DKT.stats"))
    for aseg_dtk_path in aseg_dtk_paths:
        scan_id = aseg_dtk_path.parent.parent.name.split("-")[0].replace("IXI", "")
        scan_data = stats_to_one_line(parse_aseg_dkt_stats(aseg_dtk_path))
        loaded_scans_data[scan_id] = scan_data
    df = pd.concat(loaded_scans_data.values())
    df["Subject"] = loaded_scans_data.keys()
    df = df.reset_index(drop=True)
    return df


def stats_to_one_line(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the DataFrame with parsed stats to a single line DataFrame."""
    unstacked = df.drop(columns=["StructName"]).unstack()
    unstacked = unstacked.reset_index()
    unstacked["level_1"] = unstacked["level_1"].map(lambda x: df["StructName"].iloc[x])
    unstacked.index = unstacked["level_1"] + "_" + unstacked["level_0"]
    unstacked = unstacked.drop(columns=["level_0", "level_1"])
    return unstacked.transpose()


if __name__ == "__main__":
    IXI_FastSurfer_path = r"/home/dpolak/DSC_TUTORIAL/IXI_output_stats_and_final_scans"
    IXI_demographic_path = r"/home/dpolak/DSC_TUTORIAL/IXI.xls"
    output_path = r"ixi_tabular.csv"

    df_volumetric = parse_directories(IXI_FastSurfer_path)
    df_volumetric["Subject"] = df_volumetric["Subject"].astype(int)
    df_volumetric.to_csv(output_path)

    df_demographic = pd.read_excel(IXI_demographic_path)
    df_demographic = df_demographic.rename(columns={"IXI_ID": "Subject"})
    df_demographic["Subject"] = df_demographic["Subject"].astype(int)

    df_joined = df_volumetric.merge(df_demographic, on="Subject")
    df_joined.to_csv(output_path)
