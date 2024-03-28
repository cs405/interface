from share4v.model.builder import load_pretrained_model
import os
import json
from tqdm import tqdm
from share4v.mm_utils import get_model_name_from_path
from share4v.eval.run_share4v import eval_model
from share4v.utils import disable_torch_init

import json
import os
import io
import zipfile
class Result:
    def __init__(self, id, prompt,value):
        self.id = id
        self.prompt = prompt
        self.value = value


def result_encoder(obj):
    if isinstance(obj, Result):
        return {'id': obj.id,  'prompt':obj.prompt, 'ShareGPT4V': obj.value}
    return json.JSONEncoder.default(obj)

from flask import Flask, request

app = Flask(__name__)

@app.route('/OCR_shareGPT4V', methods=['GET','POST'])
def fun():
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

    model_path = "./ckpt/ShareGPT4V-13B/"

    output_path = 'outputs/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, "en_test_shareGPT4V_False1.json")
    if (os.path.exists(output_path)):
        os.remove(output_path)

    # filename = os.path.basename(json_path)
    # output_file = os.path.join(output_path, filename)
    # output_lst = []
    model_name = get_model_name_from_path(model_path)
    print(model_name)

    # Model
    disable_torch_init()

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name)

    prompt = "Identify and transcribe all text visible in the image. Just Separate each recognized word with a space. Do not output text content that does not appear on the image."

    # prompt = "List all words visible in the image. Just Separate each recognized word with a space. Do not output text content that does not appear on the image."

    # prompt='Identify and transcribe all text visible in the image. Please extract and list out the text content clearly, ensuring accuracy in transcription. Exclude any text not appearing in the image.'
    if os.path.exists(image_folder):
        print(f"{image_folder} path is not exist")
        ans={}
        for filename in os.listdir(image_folder):
            filename_path = os.path.join(image_folder, filename)
            args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": model_name,
                "query": prompt,
                "conv_mode": None,
                "image_file": filename_path,
                "sep": ",",
                "temperature": 1,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()

            outputs = eval_model(args, tokenizer, model, image_processor, context_len)
            res = Result(filename, prompt, outputs)
            with open(output_path, "a", encoding="utf8") as file:
                json.dump(result_encoder(res), file, ensure_ascii=False, indent=4)
            ans[filename_path] = outputs
    # 将列表转换为 JSON 格式的字符串
        json_data = json.dumps(ans, ensure_ascii=False)

        # 将 JSON 字符串写入文件
        with open("data.json", "w") as file:
            file.write(json_data)
        # with open("data.json", "w") as file:
        #     json.dump(json_data, file, ensure_ascii=False, indent=4)
        return {'result': json_data}
    else:
        return {'error':'the file path of incoming has a problem,please check it'}
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5024)
