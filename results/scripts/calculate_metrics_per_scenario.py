import json
import os
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_dictionaries(csv_folder):
    dicts = {}
    files = {'SID': 'SID_list.csv', 'IID': 'IID_list.csv', 'EID': 'EID_list.csv', 'AID': 'AID_list.csv'}
    for key, filename in files.items():
        path = os.path.join(csv_folder, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                dicts[key] = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].astype(str)).to_dict()
            except: pass
    return dicts

def decode_scenario(scenario_id, lookup_dicts):
    """Translates '1010s1120s...' into 'Nursery - Mix Housing...'"""
    if not lookup_dicts: return scenario_id
    parts = scenario_id.split('s')
    if len(parts) < 4: return scenario_id
    
    sid = lookup_dicts.get('SID', {}).get(parts[0], parts[0])
    eid = lookup_dicts.get('EID', {}).get(parts[2], parts[2])
    
    return f"{sid} | {eid}"

def evaluate_per_scenario(ground_truth_path, predictions_path, output_path, csv_dir=None):
    
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)

    filename_to_id = {img['file_name'].split('/')[-1]: img['id'] for img in gt_data['images']}
    
    with open(predictions_path) as f:
        preds_data = json.load(f)

    final_preds = []
    for pred in preds_data:
        filename = pred['image_id'].split('/')[-1] if isinstance(pred['image_id'], str) else str(pred['image_id'])

        if filename in filename_to_id:
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

    scenario_groups = {}
    
    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs(img_id)[0]
        filename = img_info['file_name']
        
        scenario_id = filename.rsplit('-', 1)[0]
        
        if scenario_id not in scenario_groups:
            scenario_groups[scenario_id] = []
        scenario_groups[scenario_id].append(img_id)

    lookups = load_dictionaries(csv_dir) if csv_dir else None

    results = []
    
    print(f"Evaluating {len(scenario_groups)} unique scenarios...")
    
    for scenario_id, img_ids in scenario_groups.items():
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = img_ids
        
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
        stats = coco_eval.stats
        
        readable_name = decode_scenario(scenario_id, lookups)
        
        results.append({
            "scenario_id": scenario_id,
            "description": readable_name,
            "image_count": len(img_ids),
            "mAP_50-95": round(stats[0], 3),
            "mAP_50": round(stats[1], 3),
            "mAP_75": round(stats[2], 3),
            "AP_Medium": round(stats[4], 3),
            "AP_Large": round(stats[5], 3)
        })

    results.sort(key=lambda x: x['mAP_50-95'])

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n--- WORST 5 SCENARIOS (Low mAP) ---")
    df_results = pd.DataFrame(results)
    print(df_results[['description', 'image_count', 'mAP_50-95']].head(5).to_string(index=False))
    print(f"\nFull breakdown saved to: {output_path}")

if __name__ == "__main__":
    gt_path = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife/test/test.json"
    
    model = "sam3"
    context = "zero_shot"
    pred_path = f"/hd2/marcos/research/repos/pig-segmentation-distill/teacher/predictions.json"
    out_path = f"/hd2/marcos/research/repos/pig-segmentation-distill/results/filtered/{model}_{context}_scenario_breakdown.json"
    
    csv_folder_path = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife/descriptions/"

    evaluate_per_scenario(gt_path, pred_path, out_path, csv_folder_path)