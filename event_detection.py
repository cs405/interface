from utils import *
import os
import base64
import requests
import json
from subprocess import Popen, PIPE, STDOUT
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtWidgets import QTableWidgetItem
import logging
from PyQt6.QtCore import QThread, pyqtSignal
from concurrent.futures import ThreadPoolExecutor


# 多线程
class WorkerThread(QThread):
    # 初始化信号为bool型，若返回数据为True在主线程中执行函数
    result_signal = pyqtSignal(bool,bool, dict, dict)
    progress_signal = pyqtSignal(int)

    # 初始化类，接收类Tab_event_detection的所有参数
    def __init__(self,url1,url2,imgpath,exist,refresh,saved_model,table,result,result2,button) -> None:
        super().__init__()
        self.url1 = url1
        self.url2 = url2
        self.imgpath = imgpath
        self.exist = exist
        self.refresh = refresh
        self.saved_model = saved_model
        self.table = table
        self.data2_list = []
        self.data3_list = []
        self.result = result
        self.result2 = result2
        self.download_button = button


        logging.basicConfig(level=logging.DEBUG)
    def Upload_zip(self, imgpath=None, file_exist=None, refresh=None):
        def zip_directory(folder_path, output_io):
            with zipfile.ZipFile(output_io, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, folder_path))
        image_dir = imgpath
        url = 'http://10.10.81.31:5002/upload_file'
        # 文件存在的话，files={},否则加载数据
        if (file_exist and refresh == "Yes") or (not file_exist):
            data = {'task_name': "object_detection"}
            zip_buffer = io.BytesIO()
            zip_directory(image_dir, zip_buffer)
            zip_buffer.seek(0)
            files = {
                "file": (
                    image_dir.split("/")[-1] + ".zip",
                    zip_buffer,
                    "application/zip",
                )
            }
            try:
                response = requests.post(url, files=files, data=data)
                status_code = response.status_code
                if status_code == 200:
                    response_content = response.content.decode("utf-8")
                    result = json.loads(response_content)
                    return result
                else:
                    # self.result_signal.emit(False, False, self.result, self.result2)
                    print("Request failed with status code:", status_code)
                    self.result_signal.emit(False, False, self.result, self.result2)
            except:
                self.result_signal.emit(False, False, self.result, self.result2)
    def parallel(self,imgpath,mode_url):
        image_dir = imgpath
        data_dir_name = image_dir.split("/")[-1]
        data = {"isFileExist": True, "image_file": data_dir_name}
        try:
            response = requests.post(mode_url, files={}, data=data)
            status_code = response.status_code
            # 处理响应
            if status_code == 200:
                response_content = response.content.decode("utf-8")
                result = json.loads(response_content)
                return result
            else:
                # print(response_content)
                print("Request failed with status code:", status_code)
                self.result_signal.emit(False, False, self.result, self.result2)
        except:
            self.result_signal.emit(False, False, self.result, self.result2)


    # 由于draw也是耗时操作，所以也在run函数中实现
    def draw(self, result, path):
        def bbox_iou(box1, box2):
            # 计算两个框的交集区域
            inter_area = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(
                0, min(box1[3], box2[3]) - max(box1[1], box2[1])
            )
            # 计算两个框各自的面积
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            # 计算IoU
            iou = inter_area / float(box1_area + box2_area - inter_area)
            return iou

        def remove_iou(glee_result, threshold=0.5):
            selected_boxes = []
            selected_scores = []
            selected_labels = []

            for model1_box, model1_score, model1_label in zip(
                glee_result["boxes"], glee_result["scores"], glee_result["labels"]
            ):
                is_selected = True

                for model2_box, model2_score, model2_label in zip(
                    glee_result["boxes"], glee_result["scores"], glee_result["labels"]
                ):
                    if model1_box != model2_box:
                        iou = bbox_iou(model1_box, model2_box)
                        if iou > threshold and model1_score < model2_score:
                            is_selected = False
                            break

                if is_selected:
                    selected_boxes.append(model1_box)
                    selected_scores.append(model1_score)
                    selected_labels.append(model1_label)

            glee_result["boxes"] = selected_boxes
            glee_result["scores"] = selected_scores
            glee_result["labels"] = selected_labels
            return glee_result

        def plot_boxes_to_image(image_pitcure, glee_result):
            # H, W = tgt["size"]
            labels = [
                f"{glee_result['labels'][i]}({glee_result['scores'][i]:.2f})"
                for i in range(len(glee_result["scores"]))
            ]
            boxes = np.asarray(glee_result["boxes"])
            labels = np.asarray(labels)
            assert len(boxes) == len(labels), "boxes and labels must have same length"

            draw = ImageDraw.Draw(image_pitcure)

            # draw boxes and masks
            for box, label in zip(boxes, labels):
                color = tuple(np.random.randint(0, 255, size=3).tolist())
                # draw
                x0, y0, x1, y1 = box
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

                draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

                font = ImageFont.load_default()
                if hasattr(font, "getbbox"):
                    bbox = draw.textbbox((x0, y0), str(label), font)
                else:
                    w, h = draw.textsize(str(label), font)
                    bbox = (x0, y0, w + x0, y0 + h)
                draw.rectangle(bbox, fill=color)
                draw.text((x0, y0), str(label), fill="white")
            return image_pitcure

        def plot_boxes_to_image_merge(image_pitcure, tgt):
            boxes = np.asarray(tgt["boxes"])
            labels = np.asarray(tgt["labels"])
            assert len(boxes) == len(labels), "boxes and labels must have same length"
            draw = ImageDraw.Draw(image_pitcure)
            for box, label in zip(boxes, labels):
                color = tuple(np.random.randint(0, 255, size=3).tolist())
                # draw
                x0, y0, x1, y1 = box
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
                font = ImageFont.load_default()
                if hasattr(font, "getbbox"):
                    bbox = draw.textbbox((x0, y0), str(label), font)
                else:
                    w, h = draw.textsize(str(label), font)
                    bbox = (x0, y0, w + x0, y0 + h)
                draw.rectangle(bbox, fill=color)
                draw.text((x0, y0), str(label), fill="white")
            return image_pitcure


        detect_res = result
        output_dir = path

        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        file_names = os.listdir(self.imgpath)
        output_file = []

        for i in range(len(file_names)):
            output_file.append("./" + path + "/" + file_names[i].split("/")[-1])

        for i in range(len(file_names)):
            try:
                if len(self.saved_model) == 1:
                    if self.saved_model[0] == "yolov7":
                        if file_names[i] in detect_res.keys():
                            glee_result = detect_res[file_names[i]]
                    elif self.saved_model[0] == "GLEE":
                        if file_names[i] in detect_res["glee_result"].keys():
                            glee_result = detect_res["glee_result"][file_names[i]]
                    else:
                        print(f"Key {file_names[i]} not found in detect_res")
                        continue
                else:
                    if file_names[i] in detect_res.keys():
                        glee_result = detect_res[file_names[i]]
                    else:
                        glee_result = detect_res["glee_result"][file_names[i]]

                glee_result = remove_iou(glee_result, threshold=0.5)
                image_picture = Image.open(self.imgpath + "/" + file_names[i]).convert(
                    "RGB"
                )
                image_with_box = plot_boxes_to_image(image_picture, glee_result)
                image_with_box.save(output_file[i])
            except:
                print(f"Key {file_names[i]} can not be detected")
                shutil.copy(self.imgpath + "/" + file_names[i], "./" + path + "/" + file_names[i])
                continue

        if len(self.saved_model) == 2:
            data_path = self.imgpath
            yolov7_data = self.result # yolo的json
            glee_data = self.result2
            output_dir = 'merge_glee_version2_res_0.5_v1'  # 输出路径
            integrated_results = {} # 保存整合后的结果

            ## 删除以往目录，创建新目录
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            res = {}
            for file_name in os.listdir(data_path):
                try:
                    # glee_result = glee_data[file_name]
                    if file_name in glee_data["glee_result"].keys():
                        glee_result = glee_data["glee_result"][file_name]
                    if file_name in yolov7_data.keys():
                        yolov7_result = yolov7_data[file_name]

                    iou_threshold = 0.5
                    save_iou_threshold = 0.8
                    merge_boxes = []
                    merge_scores = []
                    merge_labels = []
                    has_exist = {}

                    glee_result = remove_iou(glee_result, threshold=0.5)

                    for model1_box, model1_score, model1_label in zip(glee_result['boxes'], glee_result['scores'],
                                                                      glee_result['labels']):
                        found_match = False
                        for model2_box, model2_score, model2_label in zip(yolov7_result['boxes'], yolov7_result['scores'],
                                                                          yolov7_result['labels']):

                            iou = bbox_iou(model1_box, model2_box)

                            if iou > iou_threshold:
                                has_exist[tuple(model2_box)] = 1
                                # has_exist[tuple(mode)]
                                if model2_score > model1_score * 2:
                                    merge_boxes.append(model2_box)
                                    merge_scores.append(model2_score)
                                    merge_labels.append(
                                        model2_label)  # 这里使用模型1的标签，如果你希望使用模型2的标签，请改为 `merge_labels.append(model2_label)`
                                else:
                                    merge_boxes.append(model1_box)
                                    merge_scores.append(model1_score)
                                    merge_labels.append(model1_label)
                                found_match = True
                                # break  # 找到匹配则跳出内层循环

                        if not found_match:  # 自定义分数阈值
                            merge_boxes.append(model1_box)
                            merge_scores.append(model1_score)
                            merge_labels.append(model1_label)

                    # grounding_result=glee_result
                    for model2_box, model2_score, model2_label in zip(yolov7_result['boxes'], yolov7_result['scores'],
                                                                      yolov7_result['labels']):
                        if tuple(model2_box) not in has_exist and model2_score > save_iou_threshold:
                            merge_boxes.append(model2_box)
                            merge_scores.append(model2_score)
                            merge_labels.append(model2_label)

                    ans = {}
                    ans['boxes'] = merge_boxes
                    ans['scores'] = merge_scores
                    ans['labels'] = merge_labels
                    res[file_name] = ans

                    labels = [f"{merge_labels[i]}({merge_scores[i]:.2f})" for i in range(len(merge_scores))]
                    integrated_results[file_name] = {
                        'boxes': merge_boxes,
                        # 'scores': merge_scores,
                        'labels': labels,
                    }

                    image_picture = Image.open(os.path.join(data_path, file_name)).convert("RGB")
                    image_with_box = plot_boxes_to_image_merge(image_picture, integrated_results[file_name])
                    image_with_box.save(os.path.join(output_dir, file_name))
                except:
                    shutil.copy(data_path + "/" + file_name, "./" + output_dir + "/" + file_name)
                    continue

    def convert_nested_json(self, json_data):
        if isinstance(json_data, dict):
            return {k: self.convert_nested_json(v) for k, v in json_data.items()}
        elif isinstance(json_data, list):
            return [self.convert_nested_json(item) for item in json_data]
        elif isinstance(json_data, str):
            try:
                return self.convert_nested_json(json.loads(json_data))
            except ValueError:
                return json_data
        else:
            return json_data

    def download(self, save_json, path, models):
        image_dir = path
        # 如果不存在E:/Lenovo/save/文件夹，则创建
        try:
            # 将save_json保存到本地
            save_path = (
                os.getcwd()
                + "/save/"
                + image_dir.split("/")[-1]
                + "_"
                + models
                + ".json"
            )
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_json, f, indent=4, ensure_ascii=False)
            print("保存成功")
        except Exception as e:
            print(e)

    def run(self):
        logging.info("Thread started.")
        with ThreadPoolExecutor(max_workers=2) as executor:
            if len(self.saved_model) == 1:
                upload_result = executor.submit(self.Upload_zip, self.imgpath, self.exist, self.refresh).result()
                if upload_result['success'] == 'upload data success':
                    logging.info("Upload_zip completed.")
                    self.progress_signal.emit(25)
                    self.result = self.parallel(self.imgpath, self.url1)
                    logging.info("parallel completed.")
                    self.progress_signal.emit(50)
                    self.result = self.convert_nested_json(self.result)
                    logging.info("convert_nested_json completed.")
                    self.progress_signal.emit(75)
                    self.draw(self.result, "res")
                    logging.info("draw completed.")
                    self.progress_signal.emit(100)
                else:
                    logging.info("Upload_zip failed.")
                    # 跳出线程
                    self.result_signal.emit(False,True, self.result, self.result2)
                    # 线程结束
                    return False
            else:
                upload_result = executor.submit(self.Upload_zip, self.imgpath, self.exist, self.refresh).result()
                if upload_result['success'] == 'upload data success':
                    self.progress_signal.emit(25)
                    future1 = executor.submit(self.parallel, self.imgpath, self.url1)
                    future2 = executor.submit(self.parallel, self.imgpath, self.url2)
                    self.result, self.result2 = future1.result(), future2.result()
                    logging.info("Upload_zip completed.")
                    self.progress_signal.emit(50)
                    self.result = self.convert_nested_json(self.result)
                    self.result2 = self.convert_nested_json(self.result2)
                    logging.info("convert_nested_json completed.")
                    self.progress_signal.emit(75)
                    self.draw(self.result, "res")
                    self.draw(self.result2, "res2")
                    logging.info("draw completed.")
                    self.progress_signal.emit(100)
                else:
                    logging.info("Upload_zip failed.")
                    # 跳出线程
                    self.result_signal.emit(False,True, self.result, self.result2)
                    # 线程结束
                    return False
        self.result_signal.emit(True,True, self.result, self.result2)
        logging.info("Thread finished.")


class Tab_event_detection(TabWidget):
    def __init__(self, ui) -> None:
        super().__init__()
        # 设置组件
        self.select_filefolder_butt: QPushButton = ui.pushButton_24
        self.mode_comboBox: QComboBox = ui.comboBox_4
        # self.correct_rate: QLineEdit = ui.lineEdit
        self.message: QLineEdit = ui.lineEdit_2
        self.img_comboBox: QComboBox = ui.comboBox_5
        self.upload_button: QPushButton = ui.pushButton
        self.try_button: QPushButton = ui.pushButton_4
        self.table: QTableWidget = ui.tableWidget
        self.pre_butt: QPushButton = ui.pushButton_2
        self.next_butt: QPushButton = ui.pushButton_3
        self.reupload_comboBox: QComboBox = ui.comboBox_6
        self.download_button: QPushButton = ui.pushButton_5
        # 设置ip
        self.model = ["yolov7", "GLEE"]
        self.exist_url = self.ip + ":5002/file_is_exist"
        self.url1 = ""
        self.url2 = ""
        self.jsonpath = []
        self.imgpath = None
        self.data2_list = []
        self.data3_list = []
        self.result = {}
        self.result2 = {}
        self.exist = False
        self.saved_model = []
        self.button = False
        self.progress_bar: QProgressBar = ui.progressBar

    def input_image(self):
        pass

    def Try(self):
        paths = self.ischecked(self.img_comboBox)
        self.saved_model = self.ischecked(self.mode_comboBox)
        if len(self.saved_model) == 1:
            if self.saved_model[0] == "yolov7":
                self.url1 = self.ip + ":5003/get_yolov7_result"
            elif self.saved_model[0] == "GLEE":
                self.url1 = self.ip + ":5002/get_glee_result"
        else:
            self.url1 = self.ip + ":5003/get_yolov7_result"
            self.url2 = self.ip + ":5002/get_glee_result"
        self.imgpath = self.select_filefolder_butt.text() + "/" + paths[0]
        network_error,self.exist = TabWidget.Is_file_exist(self.imgpath, self.exist_url, 0, 0)
        if network_error:
            if self.exist:
                self.message.setText("文件在服务器中已存在")
            else:
                self.message.setText("文件在服务器中不存在")
        else:
            self.message.setText("网络连接失败")

    def set_table_page(self, pagenum):
        """在数据列表中切换到第pagenum页，一页显示5个
        Args:
            pagenum (int): 页号
        """
        # 切换到第pagenum页
        self.pagenumber = pagenum
        # 清空表格
        self.table.clearContents()
        # 显示当前页表格元素

        for row in range(5):
            if self.pagenumber * 5 + row < self.listlength:
                num = QTableWidgetItem(str(self.pagenumber * 5 + row + 1))
                idx = QTableWidgetItem(
                    self.data_list[self.pagenumber * 5 + row].split("/")[-1][:-4]
                )
                question_pixmap = QPixmap(self.data_list[self.pagenumber * 5 + row])
                # 设置图片大小
                question_pixmap = question_pixmap.scaled(
                    320, 320, Qt.AspectRatioMode.KeepAspectRatio
                )
                question_label = QLabel()
                question_label.setPixmap(question_pixmap)
                question_label.setAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )  # 设置图片居中
                answer_pixmap1 = QPixmap(self.data2_list[self.pagenumber * 5 + row])
                answer_pixmap1 = answer_pixmap1.scaled(
                    320, 320, Qt.AspectRatioMode.KeepAspectRatio
                )  # 设置图片大小
                answer_label1 = QLabel()
                answer_label1.setPixmap(answer_pixmap1)
                answer_label1.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 设置图片居中
                if len(self.saved_model) == 2 and self.pagenumber * 5 + row < len(self.data3_list) and self.pagenumber * 5 + row < len(self.data4_list):
                    answer_pixmap2 = QPixmap(self.data3_list[self.pagenumber * 5 + row])
                    answer_pixmap2 = answer_pixmap2.scaled(
                        320, 320, Qt.AspectRatioMode.KeepAspectRatio
                    )
                    answer_label2 = QLabel()
                    answer_label2.setPixmap(answer_pixmap2)
                    answer_label2.setAlignment(
                        Qt.AlignmentFlag.AlignCenter
                    )  # 设置图片居中
                    answer_pixmap3 = QPixmap(self.data4_list[self.pagenumber * 5 + row])
                    answer_pixmap3 = answer_pixmap3.scaled(
                        320, 320, Qt.AspectRatioMode.KeepAspectRatio
                    )
                    answer_label3 = QLabel()
                    answer_label3.setPixmap(answer_pixmap3)
                    answer_label3.setAlignment(
                        Qt.AlignmentFlag.AlignCenter
                    )

                    # 创建QTableWidgetItem对象，并设置其图标
                    self.table.setItem(row, 0, num)
                    self.table.setItem(row, 1, idx)
                    self.table.setCellWidget(row, 2, question_label)
                    self.table.setCellWidget(row, 3, answer_label1)
                    self.table.setCellWidget(row, 4, answer_label2)
                    self.table.setCellWidget(row, 5, answer_label3)
                elif len(self.saved_model) == 1 and self.saved_model[0] == "yolov7":
                    empty1 = QTableWidgetItem("")
                    empty2 = QTableWidgetItem("")
                    self.table.setItem(row, 0, num)
                    self.table.setItem(row, 1, idx)
                    self.table.setCellWidget(row, 2, question_label)
                    self.table.setCellWidget(row, 3, answer_label1)
                    self.table.setItem(row, 4, empty1)
                    self.table.setItem(row, 5, empty2)
                elif len(self.saved_model) == 1 and self.saved_model[0] == "GLEE":
                    empty1 = QTableWidgetItem("")
                    empty2 = QTableWidgetItem("")
                    self.table.setItem(row, 0, num)
                    self.table.setItem(row, 1, idx)
                    self.table.setCellWidget(row, 2, question_label)
                    self.table.setItem(row, 3, empty1)
                    self.table.setCellWidget(row, 4, answer_label1)
                    self.table.setItem(row, 5, empty2)

            else:
                # 显示空
                empty1 = QTableWidgetItem("")
                empty2 = QTableWidgetItem("")
                empty3 = QTableWidgetItem("")
                empty4 = QTableWidgetItem("")
                empty5 = QTableWidgetItem("")
                empty6 = QTableWidgetItem("")
                self.table.setItem(row, 0, empty1)
                self.table.setItem(row, 1, empty2)
                self.table.setItem(row, 2, empty3)
                self.table.setItem(row, 3, empty4)
                self.table.setItem(row, 4, empty5)
                self.table.setItem(row, 5, empty6)

    def on_combobox_item_selected(self):
        self.data_list, self.data2_list, self.data3_list, self.data4_list = [], [], [], []
        for filename in os.listdir("./res"):
            self.data2_list.append("./res" + "/" + filename)
        if len(self.saved_model) == 2:
            for filename in os.listdir("./res2"):
                self.data3_list.append("./res2" + "/" + filename)
            for filename in os.listdir("./merge_glee_version2_res_0.5_v1"):
                self.data4_list.append("./merge_glee_version2_res_0.5_v1" + "/" + filename)
        # 对两个列表进行排序
        self.data2_list.sort(key=lambda x: x.split("/")[-1])
        self.data3_list.sort(key=lambda x: x.split("/")[-1])
        self.data4_list.sort(key=lambda x: x.split("/")[-1])

        for i in range(len(self.data2_list)):
            self.data_list.append(
                self.imgpath + "/" + self.data2_list[i].split("/")[-1]
            )
        self.listlength = len(self.data2_list)
        # 初始化表格内容
        self.table.setColumnWidth(0, 41)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 325)
        self.table.setColumnWidth(3, 325)
        self.table.setColumnWidth(4, 325)
        self.table.setColumnWidth(5,325)
        self.set_table_page(0)
        self.table.resizeRowsToContents()

    def finish(self):
        self.on_combobox_item_selected()
        if len(self.saved_model) == 1:
            self.download_button.clicked.connect(
                lambda: self.download(self.result, self.imgpath, self.saved_model[0])
            )
        else:
            self.download_button.clicked.connect(
                lambda: self.download(self.result, self.imgpath, "yolov7")
            )
            self.download_button.clicked.connect(
                lambda: self.download(self.result2, self.imgpath, "glee")
            )

    def upload(self):
        # 多线程执行run函数
        self.thread = WorkerThread(
            self.url1,  # 传入url1，此时为选择一个模型，地址默认放在url1
            self.url2,
            self.imgpath,  # 传入图片路径
            self.exist,  # 传入文件在服务器中是否存在
            self.reupload_comboBox.currentText(),  # 传入是否更新
            self.saved_model,  # 传入选择的模型
            self.table,  # 传入表格
            self.result,  # 传入结果
            self.result2,
            self.download_button
        )
        # self.thread.update_table_signal.connect(self.update_table)
        self.thread.progress_signal.connect(self.update_progress_bar)
        self.thread.result_signal.connect(self.upload_finished)  # 接收返回的信号
        self.thread.start()
        # 线程结束之后，将图片显示在表格中
        self.thread.finished.connect(self.finish)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    # 接收返回的信号的函数
    def upload_finished(self, finish,network, result, result2):
        if network:
            if finish:
                self.message.setText("上传成功")
                self.result = result
                self.result2 = result2
                print("上传成功")
            else:
                self.message.setText("上传失败")
                print("上传失败")
        else:
            self.message.setText("网络连接失败")
            print("网络连接失败")

    def run_prediction_base(self):
        self.select_filefolder_butt.clicked.connect(
            lambda: self.select_Filefolder(self.img_comboBox)
        )  # 如果点击选择文件夹按钮，调用openFolderDilog函数
        self.set_model(self.model, self.mode_comboBox)
        self.reupload_comboBox.addItem("Yes")
        self.reupload_comboBox.addItem("No")
        self.table.cellClicked.connect(lambda row, col: self.input_image())
        self.pre_butt.clicked.connect(self.Pre_Page)
        self.next_butt.clicked.connect(self.Next_Page)

    def run(self):
        self.run_prediction_base()  # 专属于页面1
        self.try_button.clicked.connect(lambda: self.Try())
        self.upload_button.clicked.connect(lambda: self.upload())


def event_detection(ui):
    Tab1 = Tab_event_detection(ui)
    # Tab1.__init__(ui)
    Tab1.run()
