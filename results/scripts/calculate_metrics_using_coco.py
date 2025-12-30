import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def calculate_coco_metrics(ground_truth_path, predictions_path, output_path, model_name, trained, problematic_list=None):
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)
    
    if problematic_list:
        filtered_images = [
            img for img in gt_data['images'] 
            if not any(prob in img['file_name'] for prob in problematic_list)
        ]
        
        valid_image_ids = {img['id'] for img in filtered_images}
        
        filtered_annotations = [
            ann for ann in gt_data['annotations'] 
            if ann['image_id'] in valid_image_ids
        ]
                
        gt_data['images'] = filtered_images
        gt_data['annotations'] = filtered_annotations

        temp_gt_path = ground_truth_path.replace(".json", "_filtered_temp.json")
        with open(temp_gt_path, 'w') as f:
            json.dump(gt_data, f)
        eval_gt_path = temp_gt_path
    else:
        eval_gt_path = ground_truth_path

    filename_to_id = {img['file_name'].split('/')[-1]: img['id'] for img in gt_data['images']}

    with open(predictions_path) as f:
        preds_data = json.load(f)

    final_preds = []
    for pred in preds_data:
        filename = pred['image_id'].split('/')[-1]
        
        if filename in filename_to_id:
            pred['image_id'] = filename_to_id[filename]
            final_preds.append(pred)

    coco_gt = COCO(eval_gt_path)

    coco_dt = coco_gt.loadRes(final_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metrics = {
        "Model": f"{model_name}_{trained}",
        "mAP_50-95": stats[0],
        "mAP_50": stats[1],
        "mAP_75": stats[2],
        "AP_Medium": stats[4],
        "AP_Large": stats[5]
    }

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    if problematic_list and os.path.exists(temp_gt_path):
        os.remove(temp_gt_path)

if __name__ == "__main__":
    ground_truth_path = "/hd2/marcos/research/repos/pig-segmentation-distill/data/PigLife/test/test.json"
    avoid = ["1010s1120s2001-2s5300-1", "1010s1121s2001-2s5101-1", "1010s1121s2001-2s5101-2", "1010s1121s2001-2s5301-1",
    "1010s1121s2001-2s5302-1", "1010s1121s3001-2s5001-1"]

    # Zero-shot SAM3:
    # model = "sam3"
    # context = "zero_shot"
    # output_path = f"/hd2/marcos/research/repos/pig-segmentation-distill/results/filtered/{model}_{context}_performance.json"
    # predictions_path = f"/hd2/marcos/research/repos/pig-segmentation-distill/teacher/predictions.json"
    # calculate_coco_metrics(ground_truth_path, predictions_path, output_path, model_name=model, trained=context, problematic_list=avoid)

    # YOLO models:
    models = ["yolov8n", "yolov8s", "yolov8m"]
    contexts = ["baseline", "sam3trained"]
    for model in models:
        for context in contexts:
            output_path = f"/hd2/marcos/research/repos/pig-segmentation-distill/results/filtered/{model}_{context}_performance.json"
            predictions_path = f"/hd2/marcos/research/repos/pig-segmentation-distill/student/sizes/results/{context}/{model}/predictions.json"
            calculate_coco_metrics(ground_truth_path, predictions_path, output_path, model_name=model, trained=context, problematic_list=avoid)