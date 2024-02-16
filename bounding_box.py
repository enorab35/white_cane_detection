import cv2
import argparse
import supervision as sv
import numpy as np
from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "YOLOv8 live")
    parser.add_argument("--webcam-resolution",default = [1280,720], nargs = 2, type = int)#Augmenter la resolution de la webcam
    args = parser.parse_args()
    return args

def get_holder(cane, persons):
    # cane = [id, [coords]]
    # persons = {
    # id: [coords],
    # id: [coords]
    # }
    
    person, dist = None, [float(inf), float(inf), float(inf), float(inf)]
    for key, value in persons.items():
        diff = [a-b for a,b in zip(cane[1]-value)]
        if diff[0]<dist[0] and diff[1]<dist[1] and diff[2]<dist[2] and diff[3]<dist[3]:
            dist = diff
            person = [key,value]
    return person


def distance(detections, person):
    #A FINIR!!
    for i in range(len(person.keys())): # go through every element detected by their id (one id per element)
        x1,y1,x2,y2 = detections.xyxy[i] # get the corr coord
        if detections.id[i].item() in list(person.keys()): # the detection is already known so stored in tracker
            oldx1, oldy1, oldx2, oldy2 = person[detections.id[i].item()] # get the stored coord to compare
            # we compare the difference with 15 so we can see the diff if the person is moving slowly 
            # (maybe the case for a blind person looking for a door)
            if ((x2-x1)-(oldx2-oldx1)) > 15: # the difference is positive and big enough so the person is getting closer
                print(f"{detections.id[i].item()} s'approche")
            elif ((x2-x1)-(oldx2-oldx1)) < -15: # the difference is negative and big enough so the person is stepping back
                print(f"{detections.id[i].item()} se recule")
                person[detections.id[i].item()] = x1,y1,x2,y2 # store the new coord
    return person
    
def main():
    #Access the webcam
    args = parse_arguments()
    # frame_width,frame_height = args.webcam_resolution
    frame_width, frame_height = 500,500
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # model = YOLO("yolov8n.pt")
    model = YOLO("runs/detect/train/weights/best.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    selected_class = [0,67]
    
    tracker = {} # to track person with the coord of the bounding boxes
    
    while True:
        ret, frame = cap.read()

        result=model.track(frame, persist=True) # persist to keep the track from a frame to the next one
        # print(result)
        # Visualize the results on the frame
        annotated_frame = result[0].plot()
        detections = result[0].boxes # type Boxes object
        # print(detections)

        tracker = {id: tracker[id] for id in tracker.keys() if id in detections.id.tolist()} # delete the object not in the frame anymore
        
        # pprint(tracker)

        if detections.id:
            print("detections")
            for i in range(len(detections.id)): # go through every element detected by their id (one id per element)
                classe=detections.cls[i].item() # get the correspondant class
                x1,y1,x2,y2 = detections.xyxy[i] # get the corr coord
                conf_rate = detections.conf[i]
                print(detections.id)

                if classe == 0 and conf_rate > 0.75: # if is a human and confidence at 75%

                    if detections.id[i].item() in list(tracker.keys()): # the detection is already known so stored in tracker
                        oldx1, oldy1, oldx2, oldy2 = tracker[detections.id[i].item()] # get the stored coord to compare
                        # we compare the difference with 15 so we can see the diff if the person is moving slowly 
                        # (maybe the case for a blind person looking for a door)
                        if ((x2-x1)-(oldx2-oldx1)) > 15 and ((y2-y1)-(oldy2-oldy1)) > 15: # the difference is positive and big enough so the person is getting closer
                            print(f"{detections.id[i].item()} s'approche")
                        elif ((x2-x1)-(oldx2-oldx1)) < -15 and ((y2-y1)-(oldy2-oldy1)) < -15: # the difference is negative and big enough so the person is stepping back
                            print(f"{detections.id[i].item()} se recule")
                    tracker[detections.id[i].item()] = x1,y1,x2,y2 # store the new coord
            
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__" :
    main()