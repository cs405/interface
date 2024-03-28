import os
import json
import requests
import zipfile
import io

from PyQt6.QtGui import QPixmap, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QTableWidgetItem,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QFileDialog,
    QTextEdit,
    QCheckBox,
    QApplication,
    QWidget,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt
import io, zipfile, requests


class TabWidget:
    """页面基类"""

    def __init__(self) -> None:
        self.image_label: QLabel
        self.mode_comboBox: QComboBox
        self.language_comboBox: QComboBox
        self.answerEdit: QTextEdit
        self.upload_button: QPushButton
        self.table: QTableWidget
        self.pre_butt: QPushButton
        self.next_butt: QPushButton
        self.select_filefolder_butt: QPushButton
        self.file_comboBox: QComboBox
        self.pagenumber = 0
        self.listlength = 0
        # self.url = ""
        self.folderpath = ""
        self.model_num = 1
        self.model_list = []
        self.have_uploaded = False
        self.ip = "http://10.10.81.31"
        self.exist_url = self.ip + ":5002/file_is_exist"

    def Switch_img(self, selected_row, selected_col):
        """选中表格中某一行元素切换相应图片和集成结果"""
        picname = self.table.item(selected_row, 1)
        if picname is not None:
            # 如果 item 不为 None 再操作
            # 显示对应图片
            image_path = os.path.join(
                os.path.dirname(self.folderpath),
                "images",
                picname.text(),
            )

            pix = QPixmap(image_path)
            self.image_label.setPixmap(pix)

            # 显示集成后答案
            self.answerEdit.setText(
                self.table.item(selected_row, self.table.columnCount() - 1).text()
            )

    def set_table_format(self):
        """设置行列数量和宽高"""
        self.table.setColumnCount(self.model_num + 4)
        column_headers = ["序号", "文件名", "问题", "原答案"]
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(1, 60)
        self.table.setColumnWidth(2, 200)
        self.table.setColumnWidth(3, 300)
        for i in range(self.model_num):
            column_headers.append(self.model_list[i])
            self.table.setColumnWidth(i + 4, 300)
        self.table.setHorizontalHeaderLabels(column_headers)

        for i in range(5):
            self.table.setRowHeight(i, 150)

    def Pre_Page(self):
        """切换上一页"""
        if self.pagenumber > 0:
            self.set_table_page(self.pagenumber - 1)

    def Next_Page(self):
        """切换下一页"""
        if (self.pagenumber + 1) * 5 < self.listlength:
            self.set_table_page(self.pagenumber + 1)

    def select_Filefolder(self, QcomboBox):
        """选择数据所在文件夹"""
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.FileMode.Directory)  # 设置选择模式为选择单文件夹
        current_directory = os.path.dirname(os.path.abspath(__file__))
        fd.setDirectory(current_directory)  # 设置默认打开的文件夹为当前目录
        if fd.exec():
            folderlist = fd.selectedFiles()  # 获取选择的文件夹路径
            self.folderpath = folderlist[0]
            self.select_filefolder_butt.setText(self.folderpath)
            self.selectItem(self.select_filefolder_butt.text(), QcomboBox)

    def selectItem(self, path, QcomboBox):
        QcomboBox.clear()
        names = os.listdir(path)
        for i in range(len(names)):
            item = QStandardItem(names[i])
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
            QcomboBox.model().appendRow(item)

    def ischecked(self, QcomboBox):
        # 判断self.img_comboBox中的图片是否被选中
        names = []
        for i in range(QcomboBox.model().rowCount()):
            item = QcomboBox.model().item(i)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(item.text())
        # print('this:',names)
        return names

    def set_model(self, model, itemcomboBox):
        """
        model:列表
        """
        names = model
        itemcomboBox.clear()
        for i in range(len(names)):
            item = QStandardItem(names[i])
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
            itemcomboBox.model().appendRow(item)

    def convert_nested_json(self, json_data):
        """
        递归函数，将嵌套的JSON转换为规范的JSON格式。
        """
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

    def Is_file_exist(path, file_url, file_type_num, task_name_num):
        """
        path:传入勾选的路径，
        file_url:传入的服务器是否存在文件的url，
        File_Type:文件类型
        Task_Name:任务名称
        File_Type = ["images", "json_files"]
        Task_Name = [
            0"object_detection",
            1"Chinese_polish",
            2"Chinese_image_caption",
            3"grounding_image_caption_open",
            4"ocr_en",
            5"ocr_zh",
            6"image_caption_short",(英文)
            7'image_caption_long',(英文)
            8'grounding_image_caption_close'
        ]
        """
        File_Type = ["images", "json_files"]
        Task_Name = [
            "object_detection",
            "Chinese_polish",
            "Chinese_image_caption",
            "grounding_image_caption_open",
            "ocr_en",
            "ocr_zh",
            "image_caption_short",
            "image_caption_long",
            "grounding_image_caption_close",
        ]
        file_type = File_Type[file_type_num]
        task_name = Task_Name[task_name_num]
        # 创建内存中的ZIP文件
        image_dir = path

        url_file_exist = file_url
        data = {
            "file_type": file_type,
            "dir_name": os.path.basename(image_dir),
            "task_name": task_name,
        }
        try:
            response1 = requests.post(url_file_exist, data=data)
            status_code1 = response1.status_code

            if status_code1 == 200:
                response_content = response1.content.decode("utf-8")
                file_exist = json.loads(response_content)["result"]
            else:
                print("Request failed with status code:", status_code1)
                file_exist = False
                return False, file_exist
        except:
            return False,False

        return True,file_exist

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
        except Exception as e:
            print(e)

    def run_base(self):
        """运行基础部分"""
        self.select_filefolder_butt.clicked.connect(
            lambda: self.select_Filefolder(self.file_comboBox)
        )
        self.table.cellClicked.connect(lambda row, col: self.Switch_img(row, col))
        self.pre_butt.clicked.connect(self.Pre_Page)
        self.next_butt.clicked.connect(self.Next_Page)
