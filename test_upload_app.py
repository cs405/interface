import os
import requests
import zipfile
import io
import json
# 压缩整个目录为ZIP文件
def zip_directory(folder_path, output_io):
    with zipfile.ZipFile(output_io, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

# 创建内存中的ZIP文件
image_dir = "../../data/grounding_caption_multi_choice_m3it_coco-goi_30k"


is_update=False
data_dir_name=image_dir.split("/")[-1]

#文件存在的话，files={},否则加载数据
image_file_exist=False
if not image_file_exist:
    #存在的话，上传文件
    zip_buffer = io.BytesIO()
    zip_directory(image_dir, zip_buffer)

    # 重置buffer的位置到开始处，以便读取其内容
    zip_buffer.seek(0)
    data={'task_name':"object_detection"}
    files = {'file': (image_dir.split("/")[-1]+'.zip', zip_buffer, 'application/zip')}
    url='http://10.10.81.31:5002/upload_file'
    response = requests.post(url, files=files, data=data)
    status_code = response.status_code
    # 处理响应
    if status_code == 200:
        response_content = response.content.decode('utf-8')
        result = json.loads(response_content)
        print(result)
    else:
        # print(response_content)
        print('Request failed with status code:', status_code)
else:
    #处理多个模型的并行计算
    pass



def Upload_zip(imgpath=None, file_exist=None, refresh=None):
    def zip_directory(folder_path, output_io):
        with zipfile.ZipFile(output_io, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

    # 创建内存中的ZIP文件
    image_dir = imgpath

    url = 'http://10.10.81.31:5002/upload_file'
    # 文件存在的话，files={},否则加载数据
    if (file_exist and refresh == "Yes") or (not file_exist):
        data = {'task_name': "object_detection"}
        if refresh == "Yes":
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
                    print("Request failed with status code:", status_code)
            except Exception as e:
                print(e)



def test_Upload_zip():
    pass

