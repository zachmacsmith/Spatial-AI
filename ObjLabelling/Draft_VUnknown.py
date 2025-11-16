from inference_sdk import InferenceHTTPClient
import cv2
import os
from pathlib import Path
from datetime import datetime


class RoboflowConstructionDetector:
    def __init__(self, api_key, workspace_name, workflow_id):

        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )

        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        print("connected to roboflow")

    def detect_objects(self, image_path):

        result = self.client.run_workflow(
            workspace_name=self.workspace_name,
            workflow_id=self.workflow_id,
            images={
                "image": image_path
            },
            use_cache=True  # for 15 mins
        )
        return result

    def parse_detections(self, result):
        #   api response --> usable format
        detections = []
        try:
            #   returns list
            if isinstance(result, list) and len(result) > 0:
                output = result[0]

                #   structure: output['predictions']['predictions']
                if 'predictions' in output and isinstance(output['predictions'], dict):
                    predictions_data = output['predictions']

                    #   get actual list + process each pred
                    if 'predictions' in predictions_data:
                        predictions_list = predictions_data['predictions']

                        for pred in predictions_list:
                            detections.append({
                                'label': pred.get('class', 'unknown'),
                                'confidence': pred.get('confidence', 0),
                                'box': [
                                    pred['x'] - pred['width'] / 2,  # x1
                                    pred['y'] - pred['height'] / 2,  # y1
                                    pred['x'] + pred['width'] / 2,  # x2
                                    pred['y'] + pred['height'] / 2   # y2
                                ]
                            })
        except Exception as e:
            print(f"error parsing detections: {e}")
            import traceback
            traceback.print_exc()
        return detections

    def get_detection_summary(self, detections):
        summary = {}
        for det in detections:
            label = det['label']
            summary[label] = summary.get(label, 0) + 1
        return summary

    def annotate_image(self, image_path, detections):
        img = cv2.imread(image_path)
        #   color for each item
        color_map = {
            'person': (0, 255, 0),
            'hardhat': (0, 165, 255),
            'safety_vest': (0, 255, 255),
            'safety vest': (0, 255, 255),
            'brick': (0, 0, 255),
            'brick_trowel': (255, 0, 128),
            'brick trowel': (255, 0, 128),
        }

        for det in detections:
            label = det['label']
            conf = det['confidence']
            box = det['box']
            color = color_map.get(label.lower(), (0, 255, 0))

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)     #  make box

            text = f"{label}: {conf:.0%}"                        #  make label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            #text bg
            cv2.rectangle(
                img,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            #text
            cv2.putText(
                img,
                text,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        return img

    def process_folder(self, input_folder, output_folder, save_summary=True):
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in os.listdir(input_folder)
            if Path(f).suffix.lower() in image_extensions
        ]
        if not image_files:
            print(f"\n no images found in {input_folder}")
            return
        print(f"\n  {len(image_files)} images to be processed")

        total_detections = 0
        all_detections = {}
        processing_summary = []

        for idx, image_file in enumerate(image_files, 1):
            input_path = os.path.join(input_folder, image_file)

            name, ext = os.path.splitext(image_file)
            output_filename = f"{name}_detected{ext}"
            output_path = os.path.join(output_folder, output_filename)

            try:
                print(f"\n[{idx}/{len(image_files)}] Processing: {image_file}")
                result = self.detect_objects(input_path)        #   RUN API
                detections = self.parse_detections(result)
                summary = self.get_detection_summary(detections)
                num_objects = len(detections)
                total_detections += num_objects
                for label, count in summary.items():
                    all_detections[label] = all_detections.get(label, 0) + count

                if num_objects > 0:
                    print(f"detected {num_objects} object(s):")
                    for label, count in sorted(summary.items()):
                        print(f"    - {label}: {count}")
                else:
                    print(f" no objects detected")
                annotated_img = self.annotate_image(input_path, detections)

                cv2.imwrite(output_path, annotated_img) #   SAVE FILE
                print(f" saved: {output_filename}")

                processing_summary.append({
                    'filename': image_file,
                    'detections': summary,
                    'total': num_objects
                })

            except Exception as e:
                print(f"error: {e}")

        # Final summary
        print("PROCESSING COMPLETE")
        print("=" * 67)
        print(f"total objects detected: {total_detections}")

        if all_detections:
            for label, count in sorted(all_detections.items(), key=lambda x: x[1], reverse=True):
                print(f"   {label}: {count}")
        print(f"\n results saved to: {output_folder}")
        print("=" * 67)

        # summary in a text file
        if save_summary:
            summary_path = os.path.join(output_folder, 'detection_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 67)
                f.write(f"total images: {len(image_files)}\n")
                f.write(f"total objects: {total_detections}\n\n")
                f.write("OVERALL DETECTIONS:\n")
                f.write("-" * 67)
                for label, count in sorted(all_detections.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{label}: {count}\n")

                f.write("DETAILED RESULTS\n")
                f.write("=" * 67)

                for item in processing_summary:
                    f.write(f"image: {item['filename']}\n")
                    f.write(f"total: {item['total']}\n")
                    if item['detections']:
                        for label, count in item['detections'].items():
                            f.write(f"  - {label}: {count}\n")
                    f.write("\n")


if __name__ == "__main__":

    # config
    API_KEY = "P8N1UKArr5reyZzEoDZJ"
    WORKSPACE_NAME = "rae-drepe"
    WORKFLOW_ID = "find-brick-trowels-bricks-gloves-hard-hats-and-safety-vests"

    INPUT_FOLDER = '/Users/zeyadeo/Desktop/hack'
    OUTPUT_FOLDER = '/Users/zeyadeo/Desktop/hack_output'


    # run this!!!!
    # pip install inference-sdk opencv-python


    detector = RoboflowConstructionDetector(
        api_key=API_KEY,
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID
    )

    detector.process_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        save_summary=True
    )
