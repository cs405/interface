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
# ../datasets/suzhou/ocr_zh_230920_381k
image_dir = "/data1/xyj/datasets/suzhou/images/ocr_zh_230920_381k"

# 构造要发送的数据
is_update=False
data_dir_name=image_dir.split("/")[-1]

#判断文件是否更新，更新的话默认服务器中的文件不存在，否则去发起请求验证文件是否存在
if is_update:
    file_exist=False
else:
    url_file_exist = 'http://10.10.81.31:5002/file_is_exist'
    #file_type: images或jsons
    #task_name: "object_detection" 等别的任务名
    data = {'file_type':"images",'dir_name': image_dir.split("/")[-1],'task_name':"ocr_zh"}
    response1 = requests.post(url_file_exist, data=data)
    status_code1 = response1.status_code
    if status_code1 == 200:
        response_content = response1.content.decode('utf-8')
        file_exist = json.loads(response_content)['result']
    else:
        print('Request failed with status code:', status_code1)
        file_exist=False


#文件存在的话，files={},否则加载数据
if file_exist:
    data={'isFileExist':True,'image_file':data_dir_name}
    files={}
else:
    zip_buffer = io.BytesIO()
    zip_directory(image_dir, zip_buffer)

    # 重置buffer的位置到开始处，以便读取其内容
    zip_buffer.seek(0)
    data={'isFileExist':False}
    files = {'file': (image_dir.split("/")[-1]+'.zip', zip_buffer, 'application/zip')}
# 发起POST请求
url = 'http://10.10.81.31:5022/OCR_paddleOCR_zh'  # 替换成实际的Docker服务端地址和端口号
response=requests.post(url,files=files,data=data)

status_code = response.status_code
# 处理响应
if status_code == 200:
    response_content = response.content.decode('utf-8')
    result = json.loads(response_content)
    print(result["result"])
else:
    # print(response_content)
    print('Request failed with status code:', status_code)