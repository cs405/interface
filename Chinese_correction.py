from utils import *
import logging
from PyQt6.QtCore import QThread, pyqtSignal


# 多线程
class WorkerThread(QThread):
    # 初始化信号为bool型，若返回数据为True在主线程中执行函数
    result_signal = pyqtSignal(bool)

    # 初始化类，接收类Tab_Chinese_correction的所有参数
    def __init__(self, url, exist_url, model_list, json_list, folderpath, isupdate):
        super().__init__()
        self.url = url
        self.exist_url = exist_url
        self.model_list = model_list
        self.folderpath = folderpath
        self.json_list = json_list
        self.isupdate = isupdate
        self.download_list = {}
        self.ret_data_list = {}
        self.pre_data_list = []

        logging.basicConfig(level=logging.DEBUG)

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

    def upload_json(self, json_path, file_exist, url):
        """将数据上传到服务器并解析返回结果，获取ret_data_list"""
        data = {"isFileExist": file_exist}

        if not file_exist:
            # Scenario 1: Upload the file
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            json_io = io.BytesIO(json.dumps(json_data).encode())

            # 重置buffer的位置到开始处，以便读取其内容
            json_io.seek(0)

            files = {
                "json_file": (
                    os.path.basename(json_path),
                    json_io,
                    "application/json",
                )
            }
        else:
            # Scenario 2: Server already has the file
            data["json_file"] = (
                "shared_data/Chinese_polish/json_files/" + os.path.basename(json_path)
            )
            files = {}

        response = requests.post(url, files=files, data=data)
        status_code = response.status_code

        # 处理响应
        if status_code == 200:
            response_content = response.content.decode("utf-8")
            result = json.loads(response_content)
            # print(result)
        else:
            print("Request failed with status code:", status_code)
            response_content = response.content.decode("utf-8")
            print(response_content)

        return response

    # 重写run函数
    def run(self):
        logging.info("Thread started.")
        # url

        url = self.url

        # 初始化返回列表
        for model in self.model_list:
            self.ret_data_list[model] = []

        # 原问题答案
        self.pre_data_list = []

        # 循环请求json
        for json_name in self.json_list:

            json_path = self.folderpath + "/" + json_name
            print(json_path)
            print(url)

            if self.isupdate:
                # 需要更新时默认不存在
                file_exist = False
            else:
                # 不确定更新时看服务器上是否存在
                file_exist = TabWidget.Is_file_exist(json_path, self.exist_url, 1, 1)

            response = self.upload_json(json_path, file_exist, url)
            response = response.json()

            # 存放备用下载
            self.download_list[json_name] = response
            # 读取原json并显示
            self.pre_data_list += self.read_json_file(json_path)

            # 将结果合并
            for model in self.model_list:
                for data in response[model]:
                    data["picname"] = os.path.join(
                        json_name[0:-5], os.path.basename(data["image"])
                    )
                    self.ret_data_list[model].append(data)

        self.result_signal.emit(True)
        logging.info("Thread finished.")


class Tab_Chinese_correction(TabWidget):
    def __init__(self, ui) -> None:
        super().__init__()
        # 设置组件
        self.image_label: QLabel = ui.label_6
        self.upload_button: QPushButton = ui.pushButton_17
        self.download_button: QPushButton = ui.pushButton_28
        self.pre_butt: QPushButton = ui.pushButton_25
        self.next_butt: QPushButton = ui.pushButton_16
        self.select_filefolder_butt: QPushButton = ui.pushButton_26
        self.file_comboBox: QComboBox = ui.comboBox_2
        self.answerEdit: QTextEdit = ui.textEdit_6
        self.table: QTableWidget = ui.tableWidget_6
        self.update_comboBox: QComboBox = ui.comboBox_8
        self.model_list = ["result"]
        self.download_list = {}
        self.ret_data_list = {}
        self.pre_data_list = []

        # 设置ip
        self.url = self.ip + ":5211"

    def set_table_page(self, pagenum):
        """在数据列表中切换到第pagenum页，一页显示5个"""
        # 切换到第pagenum页
        self.pagenumber = pagenum

        # 请求服务后的显示，显示返回结果答案self.ret_data_list中
        for row in range(5):
            if self.pagenumber * 5 + row < self.listlength:
                dataqwen = self.ret_data_list["qwen"][self.pagenumber * 5 + row]
                num = QTableWidgetItem(str(self.pagenumber * 5 + row + 1))
                picname = QTableWidgetItem(dataqwen["image"])
                question = QTableWidgetItem(dataqwen["conversations"][0]["value"])
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

        # self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()

    def show(self):
        # 初始化显示
        self.ret_data_list = self.thread.ret_data_list
        self.download_list = self.thread.download_list
        self.pre_data_list = self.thread.pre_data_list

        # print(self.ret_data_list)
        self.listlength = len(self.ret_data_list["result"])
        self.set_table_format()
        self.set_table_page(0)

    def Download(self):
        """保存结果到本地"""
        for json_name in self.download_list.keys():
            self.download(self.download_list[json_name], json_name[0:-5], "qwen")

    def upload_t(self):
        # 多线程执行run函数
        self.thread = WorkerThread(
            self.url,  # 传入url
            self.exist_url,
            self.model_list,
            self.ischecked(self.file_comboBox),
            self.folderpath,
            True if self.update_comboBox.currentText() == "是" else False,
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

    def run(self):
        # 运行基础组件
        self.run_base()

        # 是否更新服务器文件
        self.update_comboBox.addItems(["是", "否"])
        self.update_comboBox.setCurrentIndex(0)

        self.set_table_format()

        # 上传响应
        self.upload_button.clicked.connect(self.upload_t)

        # 下载响应
        self.download_button.clicked.connect(self.Download)


def Chinese_correction(ui):
    Tab3 = Tab_Chinese_correction(ui)
    Tab3.run()
