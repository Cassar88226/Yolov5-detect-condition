from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QIntValidator

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QPlainTextEdit, QLineEdit, QPushButton, QFileDialog, QSpinBox
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
DELAY = 10

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

# find the items by label name from detected object list
def find_item_by_label(det_list, label):
    result = []
    for det in det_list:
        if det[0] == label:
            result.append(det)
    return result
    
# check if two rectangles are intersect
# [left, top, right, bottom]
def check_intersect(rect1, rect2):
    return not (rect1[2] < rect2[0] or rect2[2] < rect1[0] or rect1[3] < rect2[1] or rect2[3] < rect2[1])
class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_log = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './best.pt'
        self.source = '0'
        self.data = './data/custom_data.yaml'
        self.project = ROOT / 'videos'
        self.name = datetime.now().strftime("%d-%m-%Y-%H-%M")
        self.video_length = 5
        self.fps = 30
    @torch.no_grad()
    def run(self, 
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):
        check_requirements(exclude=('tensorboard', 'thop'))
        weights = self.weights
        source = self.source
        data = self.data
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        if  not self.project:
            self.project = ROOT / 'videos'
        if not os.path.exists(self.project):
            os.mkdir(self.project)
        save_dir = os.path.join(self.project, self.name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()

        # Dataloader
        cudnn.benchmark = True
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        
        if source.isnumeric():
            source = int(source)
        self.vid_cap = cv2.VideoCapture(source)
        
        # Run inference
        
        model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        
        # for path, im, im0s, self.vid_cap, s in self.dataset:
        mode = "video"
        delay_pass_label = "No Pass"
        delay_number = 0
        frame_cnt = 0
        path_no = 0

        while(self.vid_cap.isOpened()):
            
            s = ''
            # Capture the video frame
            # by frame
            ret, im0s = self.vid_cap.read()
            im0s = [im0s]
            path = str(source)
        
            im0 = im0s.copy()
            img = [letterbox(x, imgsz, stride=stride, auto=pt and pt)[0] for x in im0]

            # Stack
            img = np.stack(img, 0)

            # Convert
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(img)
            
            im = im[0]

        
        
            t1 = time_sync()
            im0 = im0[0]
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), 0
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), 0

                p = Path(p)  # to Path
                if path_no == 0:
                    save_path = os.path.join(save_dir, p.name) # str(save_dir / p.name)  # im.jpg
                frame_cnt += 1
                if frame_cnt > self.fps * self.video_length * 60:
                    frame_cnt = 0
                    path_no += 1
                    file_name = p.stem + "_" + str(path_no) + p.suffix
                    save_path = os.path.join(save_dir, file_name)
                
                
                txt_path =  os.path.join(save_dir, 'labels', p.stem) if mode == 'image' else f'_{frame}'
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy()
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                # detected object list
                det_list = []
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        
                        # Get the bounding box coordinate
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        center_point = round((c1[0]+c2[0])/2), round((c1[1]+c2[1])/2)
                        
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        c = int(cls)  # integer class
                        
                        # get label name from class number
                        label = names[c] 
                        
                        # add detected object in list
                        # [label, center point, object's coordinate]                    
                        det_item = [label, center_point, list(map(int, xyxy))]                    
                        det_list.append(det_item)
                        
                        if save_img or save_crop or view_img:  # Add bbox to image
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            

                # check the image is valid(pass or no pass)
                # 1. find the motorcycle/bicycle
                #    if not found the motorcycle/bicycle, "no pass"
                # 2. find the person intersects with motorcycle/bicycle's bound box 
                #    , and center of person is over motocycle
                # 3. find the helmet
                #    , and center of helmet is over person
                
                pass_label = "No Pass"
                
                # find motorcycle/bicycle and merge them into one list
                vehicles = find_item_by_label(det_list, "motorcycle")
                bicycles = find_item_by_label(det_list, "bicycle")
                # find persons and helmets
                persons = find_item_by_label(det_list, "person")
                helmets = find_item_by_label(det_list, "helmet")
                vehicles.extend(bicycles)
                intersect_persons = []
                over_helmets = []
                
                if len(vehicles) > 0:
                    # choose only one motorcycle/bicycle
                    vehicle = vehicles[0]
                    
                    for person in persons:
                        # find persons who intersect with motorcycle/bicycle
                        if check_intersect(vehicle[2], person[2]):
                            intersect_persons.append(person)
                            # find helmet who is over person
                            for helmet in helmets:
                                h_center = helmet[1]
                                p_coord = person[2]
                                # helmet's center y value must be lesser than person's top y value
                                # and helmet' center x value must be greater than person's left and lesser than right
                                if h_center[1] < p_coord[1] and h_center[0] >= p_coord[0] and h_center[0] <= p_coord[2]:
                                    if helmet not in over_helmets:
                                        over_helmets.append(helmet)
                    
                # if person and helmet are not found, it would be "Pass"
                if len(intersect_persons) == len(over_helmets):
                    pass_label = "Pass"
                if len(intersect_persons) == 0 and len(over_helmets) == 0:
                    pass_label = "No Detection"
                delay_pass_label = pass_label
                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                if delay_number >= DELAY and delay_pass_label == "No Pass":
                    save_file = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + ".png"
                    save_file_path = os.path.join(save_dir, save_file)
                    cv2.imwrite(save_file_path, imc)
                if delay_pass_label != "No Pass":
                    delay_number = 0
                if delay_number >= DELAY:
                    delay_number = 0
                else:
                    delay_number += 1
                
                cv2.rectangle(im0, (0, 0), (230, 130), (0, 0, 0), -1)
                cv2.putText(im0, str(len(intersect_persons)) + " person", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(im0, str(len(over_helmets)) + " helmet", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(im0, pass_label, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # Stream results
                im0 = annotator.result()
                # if view_img:
                    # cv2.imshow(str(p), im0)
                    # cv2.waitKey(1)  # 1 millisecond

                
                
                # Save results (image with detections)
                if save_img:
                    if mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            fps, w, h = self.fps, im0.shape[1], im0.shape[0]
                            if not save_path.endswith(".mp4"):
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
                self.send_img.emit(im0)
                self.send_log.emit(s)
                time.sleep(1/40)
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        s = f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t
        self.send_log.emit(s)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            st = "Results saved to " + save_dir
            self.send_log.emit(st)
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


class MyWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setFixedHeight(519)
        self.setFixedWidth(980)
        self.setObjectName("MainWindow")
        # self.resize(800, 519)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background-color: rgb(55, 55, 55);")
        self.setDocumentMode(False)
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.webcam = QLabel(self.centralwidget)
        self.webcam.setGeometry(QtCore.QRect(10, 10, 591, 341))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        
        #Set Webcam Info
        self.webcam.setPalette(palette)
        self.webcam.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                "border: 3px solid black;")
        self.webcam.setText("")
        self.webcam.setObjectName("webcam")
        
        # Set Terminal Info
        self.terminal = QPlainTextEdit(self.centralwidget)
        self.terminal.setEnabled(True)
        self.terminal.setGeometry(QtCore.QRect(10, 360, 591, 151))
        self.terminal.setStyleSheet("color: rgb(255, 255, 255);\n"
                "border: 3px solid black;")
        self.terminal.setBackgroundVisible(False)
        self.terminal.setObjectName("terminal")
        
        # Set Save Path Label Info
        self.save_path = QLineEdit(self.centralwidget)
        self.save_path.setGeometry(QtCore.QRect(610, 60, 281, 23))
        self.save_path.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                "border: 3px solid black;")
        self.save_path.setObjectName("save_path")
        self.save_path_label = QLabel(self.centralwidget)
        self.save_path_label.setGeometry(QtCore.QRect(650, 20, 281, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.save_path_label.setFont(font)
        self.save_path_label.setStyleSheet("color: rgb(255, 255, 255);")
        self.save_path_label.setAlignment(QtCore.Qt.AlignCenter)
        self.save_path_label.setObjectName("save_path_label")
        
        #Set Save Path Button Info
        self.btn_save_path = QPushButton(self.centralwidget)
        self.btn_save_path.setGeometry(QtCore.QRect(900, 60, 75, 23))
        self.btn_save_path.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                "color: rgb(0, 0, 0);\n"
                "border: 3px solid black;\n"
                "border-radius:5px;")
        self.btn_save_path.setObjectName("btn_save_path")
        
        # Set Save Model Label Info
        self.btn_model_path = QPushButton(self.centralwidget)
        self.btn_model_path.setGeometry(QtCore.QRect(900, 140, 75, 23))
        self.btn_model_path.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                "color: rgb(0, 0, 0);\n"
                "border: 3px solid black;\n"
                "border-radius:5px;")
        self.btn_model_path.setObjectName("btn_model_path")
        self.model_path = QLineEdit(self.centralwidget)
        self.model_path.setGeometry(QtCore.QRect(610, 140, 281, 23))
        self.model_path.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                "border: 3px solid black;")
        self.model_path.setObjectName("model_path")
        
        #Set Start Button Info
        self.btn_start = QPushButton(self.centralwidget)
        self.btn_start.setGeometry(QtCore.QRect(630, 420, 91, 91))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.btn_start.setFont(font)
        self.btn_start.setStyleSheet("border-radius:15px;\n"
                "background-color: rgb(0, 170, 0);\n"
                "border: 3px solid black;")
        self.btn_start.setObjectName("btn_start")
        
        #Set Stop Button Info
        self.btn_stop = QPushButton(self.centralwidget)
        self.btn_stop.setGeometry(QtCore.QRect(750, 420, 91, 91))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.btn_stop.setFont(font)
        self.btn_stop.setStyleSheet("border-radius:15px;\n"
                "background-color: rgb(255, 85, 0);\n"
                "border: 3px solid black;")
        self.btn_stop.setObjectName("btn_stop")
        
        #Set Exit Button Info
        self.btn_exit = QPushButton(self.centralwidget)
        self.btn_exit.setGeometry(QtCore.QRect(870, 420, 91, 91))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.btn_exit.setFont(font)
        self.btn_exit.setStyleSheet("border-radius:15px;\n"
                "background-color: rgb(255, 0, 0);\n"
                "border: 3px solid black;")
        self.btn_exit.setObjectName("btn_exit")
        
        #Set Save Model Label Info
        self.save_model_label = QLabel(self.centralwidget)
        self.save_model_label.setGeometry(QtCore.QRect(650, 100, 281, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.save_model_label.setFont(font)
        self.save_model_label.setStyleSheet("color: rgb(255, 255, 255);")
        self.save_model_label.setAlignment(QtCore.Qt.AlignCenter)
        self.save_model_label.setObjectName("save_model_label")
        
        self.video_len = QLineEdit(self.centralwidget)
        self.video_len.setGeometry(QtCore.QRect(830, 190, 51, 23))
        self.video_len.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                     "border: 3px solid black;")
        self.video_len.setObjectName("video_len")
        self.video_len.setText("5")
        self.video_len.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        validator = QIntValidator(1, 99, self)
        self.video_len.setValidator(validator)
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(630, 190, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(880, 190, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")        
        self.setCentralWidget(self.centralwidget)
        self.terminal.setMaximumBlockCount(100)

        self.retranslateUi()
        self.connect_signal_slots()
        
    def connect_signal_slots(self):
        self.det_thread = DetThread()
        self.det_thread.source = '0'
        self.det_thread.video_length = 5
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.webcam))
        self.det_thread.send_log.connect(lambda x: self.show_log(x, self.terminal))
        self.det_thread.finished.connect(self.finish)
        self.btn_save_path.clicked.connect(self.browse_savepath)
        self.btn_model_path.clicked.connect(self.browse_modelpath)
        self.btn_start.clicked.connect(self.start_detect)
        self.btn_stop.clicked.connect(self.stop_detect)
        self.btn_exit.clicked.connect(self.exit_app)
        # self.video_len.valueChanged.connect(self.videl_len_change)
        self.btn_stop.setShortcut(QKeySequence('q'))

    @staticmethod
    def show_log(log, terminal):
        terminal.insertPlainText(log + "\n")
    
    @staticmethod
    def show_image(img_src, label):
        try:
            if len(img_src.shape) == 3:
                ih, iw, _ = img_src.shape
            else:
                ih, iw = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            img_src_ = cv2.resize(img_src, (w, h))
            # if iw > ih:
            #     scal = w / iw
            #     nw = w
            #     nh = int(scal * ih)
            #     img_src_ = cv2.resize(img_src, (nw, nh))

            # else:
            #     scal = h / ih
            #     nw = int(scal * iw)
            #     nh = h
            #     img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # def videl_len_change(self):
    #     self.det_thread.video_length = self.video_len.value()

    def browse_savepath(self):
        dirName = QFileDialog.getExistingDirectory(self, "Select Path for saving the Result")
        self.save_path.setText(dirName)
    
    def browse_modelpath(self):
        file, check = QFileDialog.getOpenFileName(self, "Select Model for detection", "", "All Files (*);;Pytorch Model (*.pt)")
        if check:
            self.model_path.setText(file)
    
    def start_detect(self):
        self.det_thread.project = self.save_path.text()
        self.det_thread.weights = self.model_path.text() if self.model_path.text() else "best.pt"
        try:
            self.det_thread.video_length = int(self.video_len.text())
        except:
            self.det_thread.video_length = 5
        print(self.det_thread.video_length)
        
        self.det_thread.name = datetime.now().strftime("%d-%m-%Y-%H-%M")
        self.det_thread.start()
    
    def finish(self):
        self.webcam.clear()
    
    def stop_detect(self):
        if hasattr(self.det_thread, 'vid_cap'):
                if self.det_thread.vid_cap:
                    self.det_thread.vid_cap.release()
        self.det_thread.quit()
        
    
    def exit_app(self):
        if hasattr(self.det_thread, 'vid_cap'):
                if self.det_thread.vid_cap:
                    self.det_thread.vid_cap.release()
        self.det_thread.quit()
        time.sleep(3)
        self.close()
    

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        #Set UI Info
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        #Set Icon for the UI
        #self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.save_path_label.setText(_translate("MainWindow", "Select Path for saving the Result"))
        self.save_model_label.setText(_translate("MainWindow", "Select Model for detection"))
        self.btn_save_path.setText(_translate("MainWindow", "SELECT"))
        self.btn_model_path.setText(_translate("MainWindow", "SELECT"))
        self.btn_start.setText(_translate("MainWindow", "START\n"
            "DETECTION"))
        self.btn_stop.setText(_translate("MainWindow", "STOP\n"
            "DETECTION"))
        self.btn_exit.setText(_translate("MainWindow", "EXIT"))
        self.label_3.setText(_translate("MainWindow", "Video Length to save:"))
        self.label_4.setText(_translate("MainWindow", "Minutes"))


if __name__ == "__main__":
    
    import sys
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())
