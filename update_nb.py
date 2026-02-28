import json
import os

path = "notebooks/JEM_Execution_Pipeline.ipynb"
if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

with open(path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        new_source = []
        changed = False
        for line in source:
            # Training update
            if "get_dataloaders(df_train, df_val, batch_size=256)" in line:
                line = line.replace(
                    "get_dataloaders(df_train, df_val, batch_size=256)",
                    'get_dataloaders(df_train, df_val, batch_size=256, artifact_dir="../models/artifacts")',
                )
                changed = True

            # Inference imports update
            if (
                "from src.model.jem.data_utils import get_inference_dataloader, get_feature_cols"
                in line
            ):
                line = line.replace(
                    "get_inference_dataloader, get_feature_cols",
                    "get_inference_dataloader, load_feature_cols",
                )
                changed = True

            # Inference call update
            if "feature_cols = get_feature_cols(test_df)" in line:
                line = line.replace(
                    "feature_cols = get_feature_cols(test_df)",
                    'feature_cols = load_feature_cols("../models/artifacts")',
                )
                changed = True

            new_source.append(line)

        if changed:
            cell["source"] = new_source

with open(path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
