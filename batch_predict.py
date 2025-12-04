
import argparse, joblib, pandas as pd, numpy as np, sys
from pathlib import Path

def load_model_and_meta(model_path="final_calibrated_model.joblib", meta_path="model_meta.joblib"):
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    cols = meta['original_cols']
    dtypes = meta.get('original_dtypes', {})
    threshold = float(meta.get('threshold', 0.5))
    return model, cols, dtypes, threshold

def align_and_cast_df(df_in, original_cols, original_dtypes, fill_value=np.nan):
    df = df_in.copy()
    for c in original_cols:
        if c not in df.columns:
            df[c] = fill_value
    df = df.loc[:, original_cols]
    for c in original_cols:
        try:
            dtype = original_dtypes.get(c, None)
            if dtype is None: continue
            if pd.api.types.is_integer_dtype(dtype):
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
            elif pd.api.types.is_float_dtype(dtype):
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
            elif pd.api.types.is_bool_dtype(dtype):
                df[c] = df[c].map({0: False,1: True}).fillna(df[c])
                df[c] = df[c].astype('boolean')
            else:
                df[c] = df[c].astype(object)
        except Exception:
            pass
    return df

def batch_predict(input_csv, output_csv, model_path, meta_path, chunk_size=None):
    model, original_cols, original_dtypes, threshold = load_model_and_meta(model_path, meta_path)
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    if chunk_size is None:
        df_in = pd.read_csv(input_csv)
        df = align_and_cast_df(df_in, original_cols, original_dtypes)
        proba = model.predict_proba(df)[:,1]
        pred = (proba >= threshold).astype(int)
        out = df_in.copy()
        out['pred'] = pred
        out['proba'] = proba
        out.to_csv(output_csv, index=False)
        print(f"Wrote predictions to {output_csv} (rows: {len(out)})")
    else:
        out_chunks = []
        for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
            df = align_and_cast_df(chunk, original_cols, original_dtypes)
            proba = model.predict_proba(df)[:,1]
            pred = (proba >= threshold).astype(int)
            chunk['pred'] = pred
            chunk['proba'] = proba
            out_chunks.append(chunk)
        out_full = pd.concat(out_chunks, axis=0)
        out_full.to_csv(output_csv, index=False)
        print(f"Wrote predictions to {output_csv} (rows: {len(out_full)})")

def parse_args():
    p = argparse.ArgumentParser(description="Batch predict with saved heart-disease model")
    p.add_argument("--input", "-i", required=True, help="Input CSV file path (patients rows)")
    p.add_argument("--output", "-o", required=True, help="Output CSV file path (will be overwritten)")
    p.add_argument("--model", default="final_calibrated_model.joblib", help="Saved model file (joblib)")
    p.add_argument("--meta", default="model_meta.joblib", help="Saved model metadata (joblib)")
    p.add_argument("--chunksize", type=int, default=None, help="If input is huge, provide chunk size (rows)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        batch_predict(args.input, args.output, args.model, args.meta, chunk_size=args.chunksize)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        raise
