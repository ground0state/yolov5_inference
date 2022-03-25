import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
class_names = model.names

# Video
frameWidth = 640
frameHeight = 360

# Video Source
cap = cv2.VideoCapture('./sample.mp4')
# cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()

    img = cv2.resize(img, (frameWidth, frameHeight))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model([img_rgb], size=640)
    bboxes = results.xyxy[0].numpy()  # 1枚目の画像の推論結果

    for bbox in bboxes:
        box, score, cl = bbox[:4], bbox[4], int(bbox[5])
        x1, y1, x2, y2 = box
        top = max(0, int(y1))
        left = max(0, int(x1))
        right = min(img.shape[1], int(x2))
        bottom = min(img.shape[0], int(y2))
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, '{0} {1:.2f}'.format(class_names[cl], score),
                    (left, top - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)

    cv2.imshow('Video', img)

    # qを押すと止まる。
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
