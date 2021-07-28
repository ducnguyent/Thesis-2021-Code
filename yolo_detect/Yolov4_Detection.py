import numpy as np
import cv2
import time
import os
from datetime import datetime

Color_map = {
    "close": (255, 0, 0),
    "open": (0, 255, 0),
    "bad_result": (10, 0, 250)
}

c1 = 1 + np.sqrt(3)  # x3 area of gt close # = 2/(sqrt(upsize)-1)
c2 = 2 / (np.sqrt(2.3) - 1)  # x2.3 area of gt open


class Detect_yolo():
    def __init__(self, weights, cfg, names, yolo_confidence_thresh, IMAGE_PATH, PATH_TO_RESULT_IMAGES_DIR, day,
                 pp=True, pa_show=False):
        super(Detect_yolo, self).__init__()
        self.weights = weights
        self.cfg = cfg
        self.names = names

        self.thresh = yolo_confidence_thresh
        self.image_path = IMAGE_PATH
        self.outputdir = PATH_TO_RESULT_IMAGES_DIR
        self.pa_show = pa_show
        self.pp = pp
        self.day = day

        self.yolov4_detector = self.yolov4_detector_init()

        self.yolo_path = ""
        self.code = ""
        self.score = []
        self.detection_time = ""
        self.label = []
        self.now = ""

    def yolov4_detector_init(self):
        yolo_net = cv2.dnn.readNet(self.weights, self.cfg)
        with open(self.names, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = yolo_net.getLayerNames()
        yolo_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

        return yolo_net, yolo_layers, classes

    def _cal_gt(self, file, width, height, move=False):
        boxes = []
        boxes_expand = []
        c_random_x = np.random.randint(-150, 150)
        c_random_y = np.random.randint(-150, 150)
        if not move:
            c_random_x = 0
            c_random_y = 0
        with open(file, "r", encoding="utf-8") as f:
            txt = f.readlines()
        for i in range(len(txt)):
            txt_split = txt[i].split(" ")
            w = int(float(txt_split[3]) * width)
            h = int(float(txt_split[4]) * height)
            x = int((float(txt_split[1]) - float(txt_split[3]) / 2) * width)
            y = int((float(txt_split[2]) - float(txt_split[4]) / 2) * height)
            boxes.append([x, y, x + w, y + h])
            if txt_split[0] == "0":
                boxes_expand.append([max(0, int(x - w / c1) + c_random_x), max(0, int(y - h / c1) + c_random_y),
                                     min(width, int(x + w + w / c1) + c_random_x),
                                     min(height, int(y + h + h / c1) + c_random_y)])
            elif txt_split[0] == "1":
                boxes_expand.append([max(0, int(x - w / c2) + c_random_x), max(0, int(y - h / c2) + c_random_y),
                                     min(width, int(x + w + w / c2) + c_random_x),
                                     min(height, int(y + h + h / c2) + c_random_y)])
        return boxes, boxes_expand

    def _cal_intersections(self, box1, box2):
        ixmin = max(box1[0], box2[0])
        ixmax = min(box1[2], box2[2])
        iymin = max(box1[1], box2[1])
        iymax = min(box1[3], box2[3])
        inter = (ixmax - ixmin) * (iymax - iymin)
        pred_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
        gt_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
        return inter, pred_area, gt_area

    def _use_additional_informations(self, gts_expanded, pred_box, ipp_threshold, ipg_threshold):
        ignore = 1
        single_ds_image = None
        for box in gts_expanded:
            inter, pred_area, gt_area = self._cal_intersections(box, pred_box)
            ipp = inter / pred_area  # intersection per predict area
            ipg = inter / gt_area  # intersection per ground-truth area
            if ipp > ipp_threshold and ipg > ipg_threshold:
                ignore = 0
        return ignore

    def predict(self):  # pp: post processing
        # save_txt_path is for evaluation. Set = None for running GUI
        image = cv2.imread(self.image_path)
        net, layers, classes = self.yolov4_detector
        height, width, channels = image.shape

        start = time.time()
        blob = cv2.dnn.blobFromImage(image, 0.00392, (512, 512), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(layers)
        self.detection_time = str(format(time.time() - start, '.3f'))

        class_ids = []
        confidences = []
        boxes = []
        conf = ""
        label = ""
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.thresh, 0.3)
        font = cv2.FONT_HERSHEY_PLAIN
        if self.pp:
            gts, gts_expanded = self._cal_gt(self.image_path[:-3] + "txt", width, height)
        x_list = []
        scores = []
        labels = []

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                conf = str(format(confidences[i] * 100, '.2f'))
                ignore = self._use_additional_informations(gts_expanded, (x, y, x + w, y + h), ipp_threshold=0.7,
                                                           ipg_threshold=0.25)
                if self.pp:
                    if ignore:
                        label = "bad_result"
                        cv2.rectangle(image, (x, y), (x + w, y + h), Color_map[label], 3)
                        cv2.putText(image, "BAD RESULT->EJECTED",
                                    (x + 30, y + 50), font, 2, Color_map[label], 3)
                        print("bad")
                        continue
                x_list.append(x)
                labels.append(label)
                scores.append(conf)

                cv2.rectangle(image, (x, y), (x + w, y + h), Color_map[label], 3)
                cv2.putText(image, label,
                            (x + 30, y + 50), font, 3, Color_map[label], 6)
                cv2.putText(image, str(self.day), (15, 90), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                            fontScale=3, thickness=3)
                self.now = datetime.now()
                current_time = self.now.strftime("%H:%M:%S")
                cv2.putText(image, str(current_time), (15, 180), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                            fontScale=3, thickness=3)

        if self.pa_show:
            for bb in gts_expanded:
                cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (114, 26, 180), 3)
        permutation = sorted(range(len(x_list)), key=lambda k: x_list[k])
        self.score = [scores[i] for i in permutation]
        self.label = [labels[i] for i in permutation]

        self.code = "yolo_" + self.now.strftime("%d%m%Y") + self.now.strftime("%H%M%S")
        self.yolo_path = self.outputdir + "Image/main/" + self.code + ".jpg"
        cv2.imwrite(self.yolo_path, image, params=None)


if __name__ == "__main__":
    YOLO_WEIGHTS = "../YOLO_Material/ver4_27.5/yolov4_DS2_last.weights"
    YOLO_CFG = "../YOLO_Material/ver4_27.5/yolov4_DS2.cfg"
    NAMES = "../YOLO_Material/DS.names"
    yolov4_model = Detect_yolo(YOLO_WEIGHTS, YOLO_CFG, NAMES, 0.7, 'E:/THESIS/test2/110_close_diagonal_227.jpg',
                               '../output/', "28 28 28", pp=True, pa_show=True)
    yolov4_model.predict()
