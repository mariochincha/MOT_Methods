# metrics_results.py

from utils.evaluator import MOTEvaluator

GT_FILE = (
    "VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt"
)

METHODS = [

    (
        "Method 1 (Baseline): YOLO11 + SORT",
        "output_method1.txt"
    ),

    (
        "Method 2 (Efficiency): YOLO26 + ByteTrack",
        "output_method2.txt"
    ),

    (
        "Method 3 (Accuracy): RT-DETR + BoT-SORT",
        "output_method3.txt"
    ),

    (
        "Method 4 (Resolution): YOLO11 + SAHI + SORT",
        "output_method4.txt"
    )

]

for method_name, pred_file in METHODS:

    print("\n" + "=" * 70)
    print(method_name)
    print("=" * 70)

    evaluator = MOTEvaluator(
        gt_file=GT_FILE,
        pred_file=pred_file
    )

    evaluator.evaluate()