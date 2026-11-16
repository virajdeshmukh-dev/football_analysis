from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('input_video/test_video.mp4', save=True)

print(results[0])
print("*" * 80)

for box in results[0].boxes:
    print(box)
