from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
import os
import io
import zipfile

torch.manual_seed(1234)
class Result:
    def __init__(self, id, value):
        self.id = id
        self.value = value


def result_encoder(obj):
    if isinstance(obj, Result):
        return {'id': obj.id,  'qwen': obj.value}
    return json.JSONEncoder.default(obj)

from flask import Flask, request

app = Flask(__name__)

@app.route('/OCR_qwen_en', methods=['GET','POST'])
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

    # 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
    tokenizer = AutoTokenizer.from_pretrained("./ckpt/qwen/", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained("./ckpt/qwen/", device_map="cuda", trust_remote_code=True).eval()

    # 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
    # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)


    output_path = './outputs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, "qwen_en.json")
    if (os.path.exists(output_path)):
        os.remove(output_path)


    generationConfig = GenerationConfig(
        chat_format="chatml",
        do_sample=False,
        eos_token_id=151643,
        max_new_tokens=512,
        max_window_size=6144,
        pad_token_id=151643,
        top_k=200,
        top_p=1,
        transformers_version="4.32.0"
    )


    query_str='Enumerate the words or sentences visible in the picture. Generate the result and just Separate each recognized word with a space.Do not output text unrelated to the content in the image.'
    if os.path.exists(image_folder):
        print(f"{image_folder} path is not exist")
        ans = {}
        for filename in os.listdir(image_folder):
            filename_path=os.path.join(image_folder,filename)
            query = tokenizer.from_list_format([
                {'image': filename_path},  # Either a local path or an url
                {'text': query_str}
            ])
            response, history = model.chat(tokenizer, query=query, history=None, generation_config=generationConfig)
            print(response)
            res = Result(filename, response)
            with open(output_path, "a", encoding="utf8") as file:
                  json.dump(result_encoder(res), file, ensure_ascii=False, indent=4)
            ans[filename_path] = response
        # 将列表转换为 JSON 格式的字符串
        json_data = json.dumps(ans, ensure_ascii=False)

        # 将 JSON 字符串写入文件
        with open("data.json", "w") as file:
            json.dump(json_data, file, ensure_ascii=False, indent=4)
            # file.write(json_data)
        return {'result': json_data}
    else:
        return {'error':'the file path of incoming has a problem,please check it'}
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5023)