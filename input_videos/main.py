from ultralytics import YOLO
model = YOLO("models/ball_detector_model.pt")
# to get the YOLO model 
results = model.track("input_videos/video_1.mp4", save=True)

print (results)
print("======================")
for box in results[0].boxes:
    print(box)
# print some of the output and get a better understanding        