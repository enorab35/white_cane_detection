import cv2
import argparse
import supervision as sv
from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "YOLOv8 live")
    parser.add_argument("--webcam-resolution",default = [1280,720], nargs = 2, type = int)#Augmenter la resolution de la webcam
    args = parser.parse_args()
    return args

def main():
    #Access the webcam
    args = parse_arguments()
    # frame_width,frame_height = args.webcam_resolution
    frame_width, frame_height = 500,500
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8n.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    oldx1, oldx2, oldy1, oldy2 = 0,0,0,0
    while True:
        ret, frame = cap.read()

        result=model(frame)[0]

        detections=sv.Detections.from_ultralytics(result)
        # detections = detections.class_id==0
        frame=box_annotator.annotate(scene=frame, detections=detections)

        if any(detections.class_id==0):
            # print(detections.xyxy)
            x1,y1,x2,y2 = detections.xyxy[0]
            if oldx1!=0 and oldx2!=0 and oldy1!=0 and oldy2!=0:
                print((x2-x1)-(oldx2-oldx1))
                if (x2-x1)-(oldx2-oldx1) > 50:
                    print(f"bigger bbox\nx1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}")
                    oldx1, oldx2, oldy1, oldy2 = x1, x2, y1, y2
            else:
                oldx1, oldx2, oldy1, oldy2 = x1, x2, y1, y2
            

        # for det in detections:
        #     # Récupérer les coordonnées de la bounding box
        #     bbox = det.xyxy[0].cpu().numpy()
            
        #     # Calculer la largeur et la hauteur de la bounding box
        #     width = bbox[2] - bbox[0]
        #     height = bbox[3] - bbox[1]

        #     print(f"Largeur de la bounding box : {width}, Hauteur de la bounding box : {height}")


        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__" : 
    main()