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


file_exist = True
threshold = 0.03


task_type = 'gigo'  # image caption short = igs, image caption long = igl, image grounding open = gigo, image grounding close = gigc
task_type_to_name={'igs':"image_caption_short",'igl':'image_caption_long','gigo':'grounding_image_caption_open','gigc':'grounding_image_caption_close',}

image_dir = "/data1/la/lenovo_dataset/images/grounding_caption_refcoco_100k/"
json_path = "/data1/la/lenovo_dataset/json_files/grounding_caption_refcoco_100k.json"

# image_dir = "../suzhou/images/grounding_caption_multi_choice_m3it_coco-goi_30k"
# json_path = "../suzhou/json_files/grounding_caption_multi_choice_m3it_coco-goi_30k.json"

# is_update=False

# if is_update:
#     file_exist=False
# else:
#     url_file_exist = 'http://10.10.81.31:5002/file_is_exist'
#     #file_type: images或json_files
#     #task_name: "object_detection" 等别的任务名
#     data = {'file_type':"json_files",'dir_name': image_dir.split("/")[-1],'task_name':task_type_to_name[task_type]}
#     response1 = requests.post(url_file_exist, data=data)
#     status_code1 = response1.status_code
#     if status_code1 == 200:
#         response_content = response1.content.decode('utf-8')
#         file_exist = json.loads(response_content)['result']
#     else:
#         print('wtfaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
#         print('Request failed with status code:', status_code1)
#         file_exist=False


if file_exist == True:
    url = 'http://10.10.81.31:5987/'
    # data = {'isFileExist': file_exist, 'threshold': threshold, 'task_type': task_type_to_name[task_type], 'image_dir': image_dir,
    #         'json_path': json_path}
    data = {'isFileExist': file_exist, 'threshold': threshold, 'task_type': task_type, 'image_dir': image_dir,
            'json_path': json_path}
    files = {}
    response = requests.post(url, files=files, data=data)
    status_code = response.status_code

else:

    # 创建内存中的ZIP文件

    zip_buffer = io.BytesIO()
    zip_directory(image_dir, zip_buffer)
    zip_buffer.seek(0)

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # if task_type=='gigc':
    #     #获取图片中红框的坐标
    #     data = {'isFileExist': False, 'task_type': task_type_to_name[task_type]}
    #     files = {'file': (image_dir.split("/")[-1] + '.zip', zip_buffer, 'application/zip')}

    #     url = 'http://10.10.81.31:5004/get_red_box_position'
    #     response = requests.post(url, files=files, data=data)
    #     status_code = response.status_code
    #     # 处理响应
    #     try:
    #         if status_code == 200:
    #             response_content = response.content.decode('utf-8')
    #             result = json.loads(response_content)
    #             # 每张图片中红框的检测结果
    #             result = result['result']
    #             for item in json_data:
    #                 item_name = item['image'].split('/')[-1]

    #                 bbox = result[item_name]['boxes']
    #                 if bbox:
    #                     item['bbox'] = bbox
    #     except Exception as e:
    #         raise Exception(f'Request failed with status code:{status_code},failed to get red box result ')


    json_io = io.BytesIO(json.dumps(json_data).encode())

    # 重置buffer的位置到开始处，以便读取其内容
    json_io.seek(0)

    dataset_name = os.path.basename(json_path).split('.')[0]
    # 发送ZIP文件
    url = 'http://10.10.81.31:5987/'
    files = {'file': (f'{dataset_name}.zip', zip_buffer, 'application/zip'),
             'json_file': (os.path.basename(json_path), json_io, 'application/json')}
    data = {'isFileExist': file_exist, 'threshold': threshold, 'task_type': task_type}
    response = requests.post(url, files=files, data=data)
    status_code = response.status_code

# 处理响应
if status_code == 200:
    response_content = response.content.decode('utf-8')
    result = json.loads(response_content)
    print(result)
    print(file_exist)
else:
    print('Request failed with status code:', status_code)
