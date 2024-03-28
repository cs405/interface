from utils import *
import logging
from PyQt6.QtCore import QThread, pyqtSignal


# 多线程
class WorkerThread(QThread):
    # 初始化信号为bool型，若返回数据为True在主线程中执行函数
    result_signal = pyqtSignal(bool)

    # 初始化类，接收类Tab_Chinese_correction的所有参数
    def __init__(
        self,
        url,
        exist,
        model_list,
        json_list,
        folderpath,
        threshold,
        is_open,
    ):
        super().__init__()
        self.url = url
        self.exist = exist
        self.model_list = model_list
        self.folderpath = folderpath
        self.json_list = json_list
        self.download_list = {}
        self.ret_data_list = {}
        self.pre_data_list = []
        self.threshold = threshold / 100
        self.is_open = is_open

        logging.basicConfig(level=logging.DEBUG)

    def upload_zip_json_grounding(self, image_dir, json_path, url, task_type):

        def zip_directory(folder_path, output_io):
            with zipfile.ZipFile(output_io, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, folder_path))

        if self.exist == True:
            # url = "http://10.10.81.31:5987/"
            data = {
                "isFileExist": self.exist,
                "threshold": self.threshold,
                "task_type": task_type,
                "image_dir": image_dir,
                "json_path": json_path,
            }
            files = {}
            response = requests.post(url, files=files, data=data)
            status_code = response.status_code

        else:
            # 创建内存中的ZIP文件
            zip_buffer = io.BytesIO()
            zip_directory(image_dir, zip_buffer)
            zip_buffer.seek(0)

            with open(json_path, "r") as f:
                json_data = json.load(f)

            json_io = io.BytesIO(json.dumps(json_data).encode())

            # 重置buffer的位置到开始处，以便读取其内容
            json_io.seek(0)

            dataset_name = os.path.basename(json_path).split(".")[0]
            # 发送ZIP文件
            # url = "http://10.10.81.31:5987/"
            files = {
                "file": (f"{dataset_name}.zip", zip_buffer, "application/zip"),
                "json_file": (os.path.basename(json_path), json_io, "application/json"),
            }
            data = {
                "isFileExist": self.exist,
                "threshold": self.threshold,
                "task_type": task_type,
            }
            print(files)
            print(data)

            response = requests.post(url, files=files, data=data)
            status_code = response.status_code

        # 处理响应
        if status_code == 200:
            response_content = response.content.decode("utf-8")
            result = json.loads(response_content)
            print(result)
            return result
        else:
            print("Request failed with status code:", status_code)

    # 重写run函数
    def run(self):
        def read_json_file(file_path):
            """
            读取JSON文件并返回解析后的Python对象。

            Parameters:
            - file_path (str): JSON文件的路径。

            Returns:
            - dict: 解析后的Python字典对象。
            """
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                return data
            except FileNotFoundError:
                print(f"文件 '{file_path}' 不存在。")
            except json.JSONDecodeError:
                print(f"无法解析文件 '{file_path}' 中的JSON数据。")

        logging.info("Thread started.")
        # 英文
        task_type = (
            "gigo" if self.is_open else "gigc"
        )  # image caption short = igs, image caption long = igl, image grounding open = gigo, image grounding close = gigc

        # 已经判断过中英文的url
        url = self.url

        # 初始化返回列表
        for model in self.model_list:
            self.ret_data_list[model] = []

        # 原问题答案
        self.pre_data_list = []
        # 循环请求json
        for json_name in self.json_list:
            image_dir = os.path.dirname(self.folderpath) + "/images/" + json_name[0:-5]
            json_path = self.folderpath + "/" + json_name
            print(image_dir)
            print(json_path)
            print(url)
            response = self.upload_zip_json_grounding(
                image_dir, json_path, url, task_type
            )
            # response = response.json()
            # 存放备用下载
            self.download_list[json_name] = response
            # 读取原json并显示
            self.pre_data_list += list(read_json_file(json_path))
            # 将结果合并
            for model in self.model_list:
                for data in response[model]:
                    data["image"] = os.path.join(
                        json_name[0:-5], os.path.basename(data["image"])
                    )
                    self.ret_data_list[model].append(data)

        self.result_signal.emit(True)
        logging.info("Thread finished.")


class Tab_grounding_caption(TabWidget):
    def __init__(self, ui) -> None:
        super().__init__()
        # 设置组件
        self.image_label: QLabel = ui.label_4
        self.mode_comboBox: QComboBox = ui.comboBox_11  # 长短文本
        self.language_comboBox: QComboBox = ui.comboBox_12  # 语言
        self.value: QLineEdit = ui.lineEdit_4  # 阈值
        # self.model_combobox: QComboBox = ui.comboBox_3  # 模型
        self.upload_button: QPushButton = ui.pushButton_11
        self.download_button: QPushButton = ui.pushButton_29
        self.pre_butt: QPushButton = ui.pushButton_13
        self.next_butt: QPushButton = ui.pushButton_12
        self.select_filefolder_butt: QPushButton = ui.pushButton_10
        self.file_comboBox: QComboBox = ui.comboBox_3
        self.answerEdit: QTextEdit = ui.textEdit_4
        self.table: QTableWidget = ui.tableWidget_4
        self.update_comboBox: QComboBox = ui.comboBox_9
        self.try_butt: QPushButton = ui.pushButton_41  # 测试文件是否在服务器
        self.file_exist = False
        self.download_list = {}
        self.ret_data_list = {}
        self.pre_data_list = []

        # 设置ip
        self.url = self.ip + ":5987"

    def set_table_page(self, pagenum):
        """在数据列表中切换到第pagenum页，一页显示5个"""
        # 切换到第pagenum页
        self.pagenumber = pagenum

        # 请求服务后的显示，显示返回结果答案self.ret_data_list中
        for row in range(5):
            if self.pagenumber * 5 + row < self.listlength:
                data = self.ret_data_list["result"][self.pagenumber * 5 + row]
                num = QTableWidgetItem(str(self.pagenumber * 5 + row + 1))
                picname = QTableWidgetItem(data["image"])
                question = QTableWidgetItem(data["conversations"][0]["value"])
                answer = QTableWidgetItem(
                    self.pre_data_list[self.pagenumber * 5 + row]["conversations"][1][
                        "value"
                    ]
                )
                self.table.setItem(row, 0, num)
                self.table.setItem(row, 1, picname)
                self.table.setItem(row, 2, question)
                self.table.setItem(row, 3, answer)
                # 多个模型答案
                for i in range(self.model_num):
                    data = self.ret_data_list[self.model_list[i]][
                        self.pagenumber * 5 + row
                    ]
                    answer = QTableWidgetItem(data["conversations"][1]["value"])
                    self.table.setItem(row, i + 4, answer)

            else:
                # 显示空
                for i in range(self.model_num + 4):
                    empty = QTableWidgetItem("")
                    self.table.setItem(row, i, empty)

        # 设置居中显示
        for row in range(self.table.rowCount()):
            for column in range(self.model_num + 4):
                self.table.item(row, column).setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )

        self.table.resizeRowsToContents()
        # self.table.resizeColumnsToContents()

    def show(self):
        # 初始化显示
        self.ret_data_list = self.thread.ret_data_list
        self.download_list = self.thread.download_list
        self.pre_data_list = self.thread.pre_data_list

        # print(self.ret_data_list)
        self.listlength = len(self.ret_data_list["result"])
        self.set_table_format()
        self.set_table_page(0)

    def on_mode_selected(self):
        """切换语言和显示的模型"""
        if self.mode_comboBox.currentText() == "open":
            self.model_num = 9
            self.model_list = [
                "qwen",
                "transcorem",
                "share4v",
                "infmllm",
                "minigpt",
                "ferret1",
                "ferret2",
                "ferret3",
                "result",
            ]
        else:
            # close
            self.model_num = 7
            self.model_list = [
                "qwen",
                "transcorem",
                "otter",
                "share4v",
                "infmllm",
                "minigpt",
                "result",
            ]
            # self.set_model(self.model_list, self.model_combobox)

        # 设置表格格式
        self.set_table_format()

    def Download(self):
        for json_name in self.download_list.keys():
            self.download(
                self.download_list[json_name],
                json_name[0:-5],
                "_".join(self.model_list[0:-1]),
            )

    def upload_t(self):
        # 多线程执行run函数

        # 切换语言和模型
        self.on_mode_selected()

        # 设置url
        url = self.url

        # 传入参数
        self.thread = WorkerThread(
            url,  # 传入url
            self.file_exist if self.update_comboBox.currentText() == "否" else False,
            self.model_list,
            self.ischecked(self.file_comboBox),
            self.folderpath,
            int(self.value.text()),
            True if self.mode_comboBox.currentText() == "open" else False,
        )

        self.thread.result_signal.connect(self.upload_finished)  # 接收返回的信号
        self.thread.start()
        # 线程结束之后，将图片显示在表格中
        self.thread.finished.connect(self.show)

    def upload_finished(self, result):
        if result:
            # self.message.setText("上传成功")
            print("上传成功")
        else:
            # self.message.setText("上传失败")
            print("上传失败")

    def test_file_exist(self):
        """测试文件是否存在"""
        print("try")
        paths = self.ischecked(self.file_comboBox)
        # 目前仅检测选中的第一个文件是否在文件夹
        self.imgpath = self.select_filefolder_butt.text() + "/" + paths[0]
        if self.mode_comboBox.currentText() == "open":
            self.file_exist = TabWidget.Is_file_exist(
                self.imgpath, self.exist_url, 1, 3
            )
        else:
            # 英文短文本
            self.file_exist = TabWidget.Is_file_exist(
                self.imgpath, self.exist_url, 1, 8
            )
        if self.file_exist:
            self.answerEdit.setText("文件在服务器中已存在")
        else:
            self.answerEdit.setText("文件在服务器中不存在")

    def run(self):
        # 运行基础组件
        self.run_base()

        # 参数框
        self.mode_comboBox.addItems(["open", "close"])
        self.mode_comboBox.setCurrentIndex(0)

        self.value.setText("25")

        self.update_comboBox.addItems(["是", "否"])
        self.update_comboBox.setCurrentIndex(0)

        self.on_mode_selected()

        # 测试响应
        self.try_butt.clicked.connect(self.test_file_exist)

        # 上传响应
        self.upload_button.clicked.connect(self.upload_t)
        self.download_button.clicked.connect(self.Download)


def grounding_caption(ui):
    Tab2 = Tab_grounding_caption(ui)
    Tab2.run()
