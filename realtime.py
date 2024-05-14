import cv2
import torch
import easyocr
from ultralytics.yolo.models.yolo import Model
from ultralytics.yolo.utils.datasets import LoadImages
from ultralytics.yolo.utils.general import non_max_suppression, scale_boxes
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.plots import Annotator, colors

class DetectionPredictor:
    def __init__(self, weights='yolov8n.pt', imgsz=(640, 640), device=''):
        self.device = select_device(device)
        self.model = Model(cfg_or_weights=weights).to(self.device)
        self.imgsz = imgsz
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self, img):
        img = self.preprocess(img)
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=10)
        return pred

reader = easyocr.Reader(['en'], gpu=True)

def ocr_image(img, coordinates):
    x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
    crop_img = img[y:h, x:w]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    text = ' '.join([res[1] for res in result])
    return text

def process_frame(frame, predictor):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pred = predictor.predict(img)
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            det[:, :4] = scale_boxes((img.shape[2], img.shape[1]), det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{predictor.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors(int(cls), True), line_thickness=3)
                # OCR
                text = ocr_image(frame, xyxy)
                cv2.putText(frame, text, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors(int(cls), True), 2)
    return frame

def plot_one_box(xyxy, img, color=(128, 128, 128), label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    device = ''  # default device
    weights = 'yolov8n.pt'  # model weights
    predictor = DetectionPredictor(weights=weights, device=device)

    cap = cv2.VideoCapture(1)  # default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, predictor)
        cv2.imshow('YOLOv8 Detection', frame)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
