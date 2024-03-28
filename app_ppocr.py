from paddleocr_v1 import PaddleOCR
import json
import os
import io
import zipfile
import shutil


class Result:
    def __init__(self, id, value):
        self.id = id
        self.value = value


def result_encoder(obj):
    if isinstance(obj, Result):
        return {'id': obj.id, 'PaddleOCR': obj.value}
    return json.JSONEncoder.default(obj)


import paddle

paddle.disable_signal_handler()  # 在2.2版本提供了disable_signal_handler接口

from flask import Flask, request

app = Flask(__name__)


@app.route('/OCR_paddleOCR_en', methods=['GET', 'POST'])
def OCR_paddleOCR_en():
    file_is_exist = request.form.get("isFileExist")
    share_folder = './shared_data'
    if file_is_exist == 'False':
        # 获取上传的图片文件内容
        zip_file = request.files['file']
        image_folder = os.path.join(share_folder, 'ocr_en/images', zip_file.filename[:-4])
        if not zip_file:
            return {'error': 'No ZIP file provided'}, 400

        # 使用 zipfile 解压 ZIP 文件
        zip_buffer = io.BytesIO(zip_file.read())
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            zip_ref.extractall(image_folder)  # 指定解压目录
    else:
        image_path = request.form.get("image_file")
        image_folder = os.path.join(share_folder, 'ocr_en/images', image_path)

    output_path = 'outputs/en'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, "PaddleOCR_en.json")
    if (os.path.exists(output_path)):
        os.remove(output_path)


    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory
    if os.path.exists(image_folder):
        print(f"{image_folder} path is not exist")
        ans = {}
        for filename in os.listdir(image_folder):
            img_path = os.path.join(image_folder, filename)
            result = ocr.ocr(img_path, cls=True)
            for res in result:
                outputs = ''
                if res is not None:
                    for line in res:
                        outputs = outputs + line[1][0] + ' '
            print(outputs)
            res = Result(filename, outputs)
            with open(output_path, "a", encoding="utf8") as file:
                json.dump(result_encoder(res), file, ensure_ascii=False, indent=4)
            ans[img_path] = outputs
        # 将列表转换为 JSON 格式的字符串
        json_data = json.dumps(ans, ensure_ascii=False)

        # 将 JSON 字符串写入文件
        with open("data.json", "w") as file:
            file.write(json_data)
        return {'result': json_data}
    else:
        return {'error':'the file path of incoming has a problem,please check it'}

@app.route('/OCR_paddleOCR_zh', methods=['GET', 'POST'])
def OCR_paddleOCR_zh():
    file_is_exist = request.form.get("isFileExist")
    share_folder = './shared_data'
    if file_is_exist == 'False':
        # 获取上传的图片文件内容
        zip_file = request.files['file']
        image_folder = os.path.join(share_folder, 'ocr_zh/images', zip_file.filename[:-4])
        if not zip_file:
            return {'error': 'No ZIP file provided'}, 400

        # 使用 zipfile 解压 ZIP 文件
        zip_buffer = io.BytesIO(zip_file.read())
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            zip_ref.extractall(image_folder)  # 指定解压目录
    else:
        image_path = request.form.get("image_file")
        image_folder = os.path.join(share_folder, 'ocr_zh/images', image_path)

    output_path = 'outputs/zh'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, "PaddleOCR_zh.json")
    if (os.path.exists(output_path)):
        os.remove(output_path)


    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    if os.path.exists(image_folder):
        print(f"{image_folder} path is not exist")
        ans = {}
        for filename in os.listdir(image_folder):
            img_path = os.path.join(image_folder, filename)
            result = ocr.ocr(img_path, cls=True)
            for res in result:
                outputs = ''
                if res is not None:
                    for line in res:
                        outputs = outputs + line[1][0] + ' '
            print(outputs)
            res = Result(filename, outputs)
            with open(output_path, "a", encoding="utf8") as file:
                json.dump(result_encoder(res), file, ensure_ascii=False, indent=4)
            ans[img_path] = outputs
        # 将列表转换为 JSON 格式的字符串
        json_data = json.dumps(ans, ensure_ascii=False)

        # 将 JSON 字符串写入文件
        with open("data.json", "w") as file:
            file.write(json_data)
        return {'result': json_data}
    else:
        return {'error':'the file path of incoming has a problem,please check it'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5022)
