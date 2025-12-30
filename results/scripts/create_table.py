import json
import os

BASE_DIR = "/hd2/marcos/research/repos/pig-segmentation-distill/results"
FILTERED_DIR = os.path.join(BASE_DIR, "filtered")

TABLE_STRUCTURE = {
    "h_n": ("yolov8n_baseline_performance.json", "YOLOv8n", "3.2"),
    "h_s": ("yolov8s_baseline_performance.json", "YOLOv8s", "11.2"),
    "h_m": ("yolov8m_baseline_performance.json", "YOLOv8m", "25.9"),
    "s_n": ("yolov8n_sam3trained_performance.json", "YOLOv8n", "3.2"),
    "s_s": ("yolov8s_sam3trained_performance.json", "YOLOv8s", "11.2"),
    "s_m": ("yolov8m_sam3trained_performance.json", "YOLOv8m", "25.9"),
    "zero": ("sam3_zero_shot_performance.json", "SAM3", "---")
}

def format_value(original_decimal, filtered_decimal):
    orig = round(original_decimal * 100, 1)
    filt = round(filtered_decimal * 100, 1)
    diff = round(filt - orig, 1)
    
    if diff > 0:
        return f"{orig:.1f} \\plus{{{diff:.1f}}}"
    elif diff < 0:
        return f"{orig:.1f} \\minus{{{abs(diff):.1f}}}"
    else:
        return f"{orig:.1f}"

def get_row_data(filename):
    orig_path = os.path.join(BASE_DIR, filename)
    filt_path = os.path.join(FILTERED_DIR, filename)
    
    with open(orig_path, 'r') as f:
        orig_data = json.load(f)
    with open(filt_path, 'r') as f:
        filt_data = json.load(f)
    
    keys = ["mAP_50-95", "mAP_50", "mAP_75", "AP_Medium", "AP_Large"]
    
    row_cells = [format_value(orig_data[k], filt_data[k]) for k in keys]
    return " & ".join(row_cells)

table_rows = {key: get_row_data(info[0]) for key, info in TABLE_STRUCTURE.items()}

latex_output = f"""
\\begin{{table*}}[t]
  \\caption{{Object detection performance (COCO metrics) of YOLOv8 models. Values in parentheses denote the change after filtering the images.}}
  \\centering
  \\label{{tab:yolo_ap_metrics}}
  \\begin{{tabular}}{{c l c c c c c c}} 
  \\toprule
  \\textbf{{Annotation}} & \\textbf{{Model}} & \\textbf{{Params (M)}} & \\textbf{{$mAP$}} & \\textbf{{$AP_{{50}}$}} & \\textbf{{$AP_{{75}}$}} & \\textbf{{$AP_{{M}}$}} & \\textbf{{$AP_{{L}}$}} \\\\
  \\midrule
  \\multirow{{3}}{{*}}{{\\makecell{{Human\\\\ annotated}}}} 
    & {TABLE_STRUCTURE['h_n'][1]} & {TABLE_STRUCTURE['h_n'][2]}  & {table_rows['h_n']} \\\\
    & {TABLE_STRUCTURE['h_s'][1]} & {TABLE_STRUCTURE['h_s'][2]} & {table_rows['h_s']} \\\\
    & {TABLE_STRUCTURE['h_m'][1]} & {TABLE_STRUCTURE['h_m'][2]} & {table_rows['h_m']} \\\\
  \\midrule
  \\multirow{{3}}{{*}}{{\\makecell{{SAM3\\\\ generated}}}} 
    & {TABLE_STRUCTURE['s_n'][1]} & {TABLE_STRUCTURE['s_n'][2]}  & {table_rows['s_n']} \\\\
    & {TABLE_STRUCTURE['s_s'][1]} & {TABLE_STRUCTURE['s_s'][2]} & {table_rows['s_s']} \\\\
    & {TABLE_STRUCTURE['s_m'][1]} & {TABLE_STRUCTURE['s_m'][2]} & {table_rows['s_m']} \\\\
  \\midrule
  \\makecell{{Zero-shot\\\\ baseline}} & {TABLE_STRUCTURE['zero'][1]} & {TABLE_STRUCTURE['zero'][2]} & {table_rows['zero']} \\\\
  \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
"""

print(latex_output)

with open(f"{BASE_DIR}/metrics_table.tex", "w") as f:
    f.write(latex_output)