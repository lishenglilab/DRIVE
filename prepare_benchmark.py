import pandas as pd
import os
import sys

def prepare(crc_path, out_path, d_n, c_n):
    if not os.path.exists(out_path): os.makedirs(out_path)
    pd.read_csv(os.path.join(crc_path, "drug_depmap.csv")).head(d_n).to_csv(os.path.join(out_path, "drug.csv"), index=False)
    gene = pd.read_csv(os.path.join(crc_path, "exp_tcga_na.csv")).head(c_n)
    new_cols = list(gene.columns); new_cols[0] = ""; gene.columns = new_cols
    master_ids = gene.iloc[:, 0].values
    gene.to_csv(os.path.join(out_path, "gene.csv"), index=False)
    others = {"mu_tcga.csv": "mu.csv", "cnv_all.csv": "cnv.csv", "mi_all.csv": "mi.csv", "tcga_gsea.csv": "gsva.csv"}
    for src, dst in others.items():
        src_path = os.path.join(crc_path, src)
        if os.path.exists(src_path):
            df = pd.read_csv(src_path).head(c_n)
            df.iloc[:, 0] = master_ids[:len(df)]
            tmp_cols = list(df.columns); tmp_cols[0] = ""; df.columns = tmp_cols
            df.to_csv(os.path.join(out_path, dst), index=False)

if __name__ == "__main__":
    prepare(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))