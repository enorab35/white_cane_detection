from ultralytics import YOLO
import cv2


model = YOLO('yolov8n-pose.pt')

# video_path=0
# cap=cv2.VideoCapture(video_path)

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         results = model(frame, save=True)
#         annoted_frame = results[0].plot()
#         cv2.imshow("YOLO inference", annoted_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()

# source = "https://ultralytics.com/images/bus.jpg"
source= "D:/enora/Documents/ISEN/M2/Projet IA/white_cane_database/database/dataset/images/IMG_6447_MOV-4_jpg.rf.8c179dde871cedb2d6bf93b56878eac5.jpg"
results = model.predict(source, save=True, imgsz=320, conf=0.5)
print(results[0].masks)
print(results[0].keypoints)