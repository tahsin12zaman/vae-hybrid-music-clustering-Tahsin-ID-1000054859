
import pandas as pd
df = pd.read_csv("results_easy/embeddings/meta.csv")
# Expect columns like filename/language; if yours differ, open the csv once and adjust.
# Print columns to be safe:
print("meta columns:", df.columns.tolist())
# Try common names:
fn_col = "filename" if "filename" in df.columns else ("path" if "path" in df.columns else df.columns[0])
lang_col = "language" if "language" in df.columns else df.columns[1]
out = pd.DataFrame({
    "filename": df[fn_col].astype(str).apply(lambda x: x.split("/")[-1]),
    "language": df[lang_col].astype(str),
    "lyrics": [""] * len(df),
})
out.to_csv("data/lyrics/lyrics.csv", index=False)

