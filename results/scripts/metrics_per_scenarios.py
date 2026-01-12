import json
import os
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib
import io

TARGET_GROUPS = [
    ["1050s1132a1110s3003-3s5001"],
    ["1050s1132a1110s3003-5s5001"],
    ["1050s1132a1110s3004", "1050s1132a1112"],
    ["1010s1120", "1010s1121"],               
    ["1060s1112", "1060a1000s1110"],
    ["1020s1120", "1020s1220"],
    ["1040s1132"],
    ["1020s1121", "1020s1221"]
]

def evaluate_custom_groups(ground_truth_path, predictions_path, output_path, csv_dir=None):
    
    # 1. Load Ground Truth
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)

    # Map filenames to COCO IDs for safer lookups
    filename_to_id = {img['file_name'].split('/')[-1]: img['id'] for img in gt_data['images']}
    
    # 2. Load and Filter Predictions
    with open(predictions_path) as f:
        preds_data = json.load(f)

    final_preds = []
    for pred in preds_data:
        # Handle cases where image_id might be a full path or just filename
        filename = str(pred['image_id']).split('/')[-1]

        if filename in filename_to_id:
            # Ensure the pred uses the integer ID expected by COCO
            pred['image_id'] = filename_to_id[filename]
            final_preds.append(pred)
        elif pred['image_id'] in filename_to_id.values():
            final_preds.append(pred)

    coco_gt = COCO(ground_truth_path)

    if final_preds:
        coco_dt = coco_gt.loadRes(final_preds)
    else:
        print("No valid predictions matched GT images.")
        return

    # 3. Group Images by TARGET_GROUPS
    # Initialize buckets for each group index
    group_buckets = {i: [] for i in range(len(TARGET_GROUPS))}
    
    print("Grouping images...")
    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs(img_id)[0]
        filename = img_info['file_name']
        
        # check which group matches this filename
        for i, group_prefixes in enumerate(TARGET_GROUPS):
            # If filename starts with ANY of the prefixes in this group
            if any(filename.startswith(prefix) for prefix in group_prefixes):
                group_buckets[i].append(img_id)
                break # Assign to the first matching group and move on

    results = []

    # 4. Evaluate per Group
    print(f"Evaluating {len(TARGET_GROUPS)} Defined Groups...")
    
    for i, img_ids in group_buckets.items():
        group_prefixes = TARGET_GROUPS[i]
        group_label = f"Group {i+1}"
        
        if not img_ids:
            print(f"Skipping {group_label}: No images found.")
            continue

        # Run COCO Eval on this subset
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = img_ids
        
        # Suppress standard COCO print output to keep console clean
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
        stats = coco_eval.stats
        
        # Try to get a readable description from the first prefix in the group
        
        results.append({
            "group_name": group_label,
            "filter_ids": ", ".join(group_prefixes),
            "image_count": len(img_ids),
            "mAP_50-95": round(stats[0], 3),
            "mAP_50": round(stats[1], 3),
            "mAP_75": round(stats[2], 3),
            "AP_Medium": round(stats[4], 3),
            "AP_Large": round(stats[5], 3)
        })

    # Sort results (optional, maybe you want them in Group order, or mAP order)
    # Let's sort by mAP to see best performers
    results.sort(key=lambda x: x['mAP_50-95'])

    # 5. Save and Print
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n--- EVALUATION RESULTS BY GROUP ---")
    df_results = pd.DataFrame(results)
    # Reorder columns for display
    display_cols = ['group_name', 'filter_ids', 'image_count', 'mAP_50-95', 'mAP_50']
    print(df_results[display_cols].to_string(index=False))
    print(f"\nFull breakdown saved to: {output_path}")

if __name__ == "__main__":
    gt_path = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife/test/test.json"
    
    model = "sam3"
    context = "zero_shot"
    pred_path = "/hd2/marcos/research/repos/pig-segmentation-distill/teacher/predictions.json"
    
    # Updated filename to reflect it's grouped logic
    out_path = f"/hd2/marcos/research/repos/pig-segmentation-distill/results/filtered/{model}_{context}_custom_groups.json"
    
    csv_folder_path = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife/descriptions/"

    evaluate_custom_groups(gt_path, pred_path, out_path, csv_folder_path)