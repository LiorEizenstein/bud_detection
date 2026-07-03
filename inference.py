import argparse
from ultralytics import YOLO

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run bud detection on a directory of images')
    p.add_argument('--weights', default='runs/bud_detector/weights/best.pt',
                   help='Path to trained model weights')
    p.add_argument('--source', required=True,
                   help='Directory of images or single image path')
    p.add_argument('--output', default='inference_results',
                   help='Output directory for annotated images')
    p.add_argument('--conf', type=float, default=0.25,
                   help='Confidence threshold (default: 0.25)')
    args = p.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        save=True,
        project=args.output,
        name='output',
        conf=args.conf,
    )
    print(f"Results saved to {args.output}/output")
