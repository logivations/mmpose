import os
import json

# Paths for your annotations files.
synthetic_file = "/data/wurth_optimization_dataset/annotations/generated.json"          # New synthetic annotations.
existing_file = "/data/wurth_optimization_dataset/annotations/forklift_keypoints_train2017.json"    # Existing annotations.
combined_file = "/data/wurth_optimization_dataset/annotations/combined_annotations.json"    # Output file.

# --- Load synthetic annotations ---
with open(synthetic_file, "r") as f:
    synthetic_data = json.load(f)

# --- Load existing annotations ---
if os.path.exists(existing_file):
    with open(existing_file, "r") as f:
        existing_data = json.load(f)
else:
    # If no existing file, use empty structure.
    existing_data = {"images": [], "annotations": [], "categories": []}

# --- Extract lists ---
synthetic_images = synthetic_data.get("images", [])
synthetic_annotations = synthetic_data.get("annotations", [])
synthetic_categories = synthetic_data.get("categories", [])

existing_images = existing_data.get("images", [])
existing_annotations = existing_data.get("annotations", [])
existing_categories = existing_data.get("categories", [])

# --- Adjust IDs to avoid collisions ---
# Compute the maximum existing image and annotation IDs.
existing_image_ids = [img["id"] for img in existing_images]
existing_ann_ids = [ann["id"] for ann in existing_annotations]

# Offsets (if there are no images/annotations, use 0).
image_id_offset = max(existing_image_ids) if existing_image_ids else 0
ann_id_offset = max(existing_ann_ids) if existing_ann_ids else 0

# Adjust synthetic image IDs.
adjusted_synthetic_images = []
for img in synthetic_images:
    new_img = img.copy()
    new_img["id"] = int(new_img["id"]) + image_id_offset
    adjusted_synthetic_images.append(new_img)

# Adjust synthetic annotations.
adjusted_synthetic_annotations = []
for ann in synthetic_annotations:
    new_ann = ann.copy()
    new_ann["id"] = int(new_ann["id"]) + ann_id_offset
    new_ann["image_id"] = int(new_ann["image_id"]) + image_id_offset
    adjusted_synthetic_annotations.append(new_ann)

# --- Merge Data ---
combined_images = existing_images + adjusted_synthetic_images
combined_annotations = existing_annotations + adjusted_synthetic_annotations

# For categories, you have options:
# Option 1: Use existing categories if they match your synthetic ones.
# Option 2: Merge them (here we simply use the existing ones if present, otherwise synthetic).
combined_categories = existing_categories if existing_categories else synthetic_categories

combined_data = {
    "images": combined_images,
    "annotations": combined_annotations,
    "categories": combined_categories
}

# --- Save the Combined Annotations File ---
with open(combined_file, "w") as f:
    json.dump(combined_data, f, indent=4)

print("Combined annotations file written to:", combined_file)
