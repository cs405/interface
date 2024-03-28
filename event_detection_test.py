import requests
import json
from concurrent.futures import ThreadPoolExecutor
import os


def upload(image_dir, url):
    try:
        # 请求单个文件的结果，file_exist=True
        # 构造要发送的数据
        data_dir_name = image_dir.split("/")[-1]
        data = {"isFileExist": True, "image_file": data_dir_name}
        files = {}

        # 发起POST请求
        response = requests.post(url, files=files, data=data)

        status_code = response.status_code
        # 处理响应
        if status_code == 200:
            response_content = response.content.decode("utf-8")
            result = json.loads(response_content)
            # print(result)
            return result
        else:
            # print(response_content)
            print("Request failed with status code:", status_code)
            return {"error": f"Request failed with status code: {status_code}"}

    except Exception as e:
        # 处理其他异常情况
        print("An error occurred:", str(e))
        return {"error": f"An error occurred: {str(e)}"}

def download(save_json, image_dir, models):
    # 如果不存在save文件夹，则创建
    if not os.path.exists(os.getcwd() + "/save/"):
        os.makedirs(os.getcwd() + "/save/")
    try:
        # 将save_json保存到本地
        save_path = (
            os.getcwd() + "/save/" + image_dir.split("/")[-1] + "_" + models + ".json"
        )
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_json, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(e)


ip = "http://10.10.81.31"
url_dict = {
    "Yolov7": ip + ":5003/get_yolov7_result",
    "GLEE": ip + ":5002/get_glee_result"
}

# 模型列表
model_list = ["Yolov7", "GLEE"]


# 获取所有文件夹
image_data_dir = "./data/images"
image_dir_list = os.listdir(image_data_dir)

for image_dir in image_dir_list:
    for model in model_list:
        with ThreadPoolExecutor(max_workers=len(model_list)) as executor:
            # Define a function to be executed by each thread
            def process_model(model):
                url = url_dict[model]
                # print(os.path.basename(image_dir))
                response = upload(image_dir, url)
                result = eval(response["result"])
                return result

            # Use the ThreadPoolExecutor to concurrently execute the function for each model
            futures = {
                executor.submit(process_model, model): model for model in model_list
            }

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

            # Retrieve results and store them in self.ret_data_list
            for future in futures.items():
                try:
                    result = future.result()  # Retrieve the result of the future
                    if type(result) == type({}) and "error" in result:
                        raise Exception

                    # 将当前model生成结果保存
                    download(result, image_dir, model)
                    print(f"Success processing model {model}")

                except Exception as e:
                    # Handle exceptions if any
                    print(f"Error processing model {model}")






