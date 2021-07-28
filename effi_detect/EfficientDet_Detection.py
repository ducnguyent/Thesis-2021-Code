import cv2
import numpy as np
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from datetime import datetime


class Detect_Effi(object):
    def __init__(self, PATH_TO_MODEL, PATH_TO_LABELS, IMAGE_PATHS, PATH_TO_RESULT_IMAGES_DIR, day):
        self.path_to_labels = PATH_TO_LABELS
        self.input_dir = IMAGE_PATHS
        self.output_dir = PATH_TO_RESULT_IMAGES_DIR
        self.path_to_model = PATH_TO_MODEL
        self.detection_model = self.load_model_efficientdet()
        self.day = day

        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        self.detection_scores = None
        self.detection_boxes = None
        self.detection_classes = None

        self.now = ""
        self.eff_path = ""
        self.score = ""
        self.label = ""
        self.detection_time = ""
        self.code = ""

    def load_model_efficientdet(self):
        detection_model = tf.saved_model.load(self.path_to_model)
        return detection_model

    def predict(self):
        img = cv2.imread(self.input_dir)
        img = cv2.resize(img, (786, 786))
        img_array = np.asarray(img)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(img_array)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        start = time.time()
        # Run inference
        detections = self.detection_model(input_tensor)
        self.detection_time = str(format(time.time() - start, '.3f'))

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        self.detection_scores = detections['detection_scores']
        self.detection_classes = detections['detection_classes']
        self.detection_boxes = detections['detection_boxes']

        # draw bounding boxes and labels
        # image, path, label, score, detect_time =
        self._draw(img)

    def _draw(self, image):
        height, width, _ = image.shape
        img_path = self.input_dir
        now = datetime.now()

        for i, score in enumerate(self.detection_scores):
            if score < 0.5:
                continue

            # if background, ignore
            if self.detection_classes[i] == 0:
                continue

            self.label = str(self.category_index[self.detection_classes[i]]['name'])
            self.score = str(format(score * 100, '.2f'))
            ymin, xmin, ymax, xmax = self.detection_boxes[i]
            real_xmin, real_ymin, real_xmax, real_ymax = int(xmin * width), int(ymin * height), int(xmax * width), int(
                ymax * height)
            cv2.rectangle(image, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
            cv2.putText(image, self.label, (real_xmin, real_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                        fontScale=0.5)
            cv2.putText(image, str(self.day), (15, 45), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                        fontScale=1, thickness=2)
            self.now = datetime.now()
            current_time = str(self.now.strftime("%H:%M:%S"))
            cv2.putText(image, current_time, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                        fontScale=1, thickness=2)
            self.code = "eff_" + self.now.strftime("%d%m%Y") + self.now.strftime("%H%M%S")
            self.eff_path = self.output_dir + "Image/support/" + self.code + ".jpg"
            cv2.imwrite(self.eff_path, image, params=None)


if __name__ == "__main__":
    PATH_TO_SAVED_MODEL = '../EfficientDet_Material/test_model/new_d1/saved_model-d1-newdata-30000/'
    eff_model = Detect_Effi(PATH_TO_SAVED_MODEL, '../EfficientDet_Material/DCL_label_map.pbtxt', '/', '/', 0)
    print(eff_model.detection_model)
