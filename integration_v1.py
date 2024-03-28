import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def plot_boxes_to_image(image_pitcure, tgt):
    # H, W = tgt["size"]
    boxes = np.asarray(tgt["boxes"])
    labels = np.asarray(tgt["labels"])
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pitcure)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        # box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        # box[:2] -= box[2:] / 2
        # box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        # mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pitcure
def bbox_iou(box1, box2):
    # 计算两个框的交集区域
    inter_area = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(
        box1[1], box2[1]))

    # 计算两个框各自的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def remove_iou(glee_result,threshold=0.5):
    selected_boxes = []
    selected_scores = []
    selected_labels = []

    for model1_box, model1_score, model1_label in zip(glee_result['boxes'], glee_result['scores'],
                                                      glee_result['labels']):
        is_selected = True

        for model2_box, model2_score, model2_label in zip(glee_result['boxes'], glee_result['scores'],
                                                          glee_result['labels']):
            if model1_box != model2_box:
                iou = bbox_iou(model1_box, model2_box)
                if iou > threshold and model1_score < model2_score:
                    is_selected = False
                    break

        if is_selected:
            selected_boxes.append(model1_box)
            selected_scores.append(model1_score)
            selected_labels.append(model1_label)

    glee_result['boxes'] = selected_boxes
    glee_result['scores'] = selected_scores
    glee_result['labels'] = selected_labels
    return glee_result


# 示例用法
data_path="./data/images/grounding_caption_multi_choice_m3it_coco-goi_30k" # imgpath
yolov7_file_path = 'data_yolov7.json' # 返回的第一个json文件
glee_file_path = 'data_glee_version2.json' # 返回的第二个json文件

output_dir='merge_glee_version2_res_0.5_v1' # 输出路径
yolov7_data= read_json_file(yolov7_file_path)
glee_data=read_json_file(glee_file_path)

integrated_results={}


## 删除以往目录，创建新目录
import shutil

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

res={}

for file_name in os.listdir(data_path):
    glee_result=glee_data[file_name]
    yolov7_result=yolov7_data[file_name]

    iou_threshold = 0.5
    save_iou_threshold=0.8
    merge_boxes = []
    merge_scores = []
    merge_labels = []
    has_exist={}

    glee_result=remove_iou(glee_result,threshold=0.5)

    for model1_box, model1_score, model1_label in zip(glee_result['boxes'], glee_result['scores'],
                                                      glee_result['labels']):
        found_match = False
        for model2_box, model2_score, model2_label in zip(yolov7_result['boxes'], yolov7_result['scores'],
                                                          yolov7_result['labels']):

            iou = bbox_iou(model1_box, model2_box)

            if iou > iou_threshold:
                has_exist[tuple(model2_box)] = 1
                # has_exist[tuple(mode)]
                if model2_score>model1_score*2:
                    merge_boxes.append(model2_box)
                    merge_scores.append(model2_score)
                    merge_labels.append(model2_label)  # 这里使用模型1的标签，如果你希望使用模型2的标签，请改为 `merge_labels.append(model2_label)`
                else:
                    merge_boxes.append(model1_box)
                    merge_scores.append(model1_score)
                    merge_labels.append(model1_label)
                found_match = True
                # break  # 找到匹配则跳出内层循环

        if not found_match:  # 自定义分数阈值
            merge_boxes.append(model1_box)
            merge_scores.append(model1_score)
            merge_labels.append(model1_label)

    # grounding_result=glee_result
    for model2_box, model2_score, model2_label in zip(yolov7_result['boxes'], yolov7_result['scores'],
                                                      yolov7_result['labels']):
        if tuple(model2_box) not in has_exist and model2_score>save_iou_threshold:
            merge_boxes.append(model2_box)
            merge_scores.append(model2_score)
            merge_labels.append(model2_label)

    # merge_boxes=glee_result['boxes']
    # merge_scores=glee_result['scores']
    # merge_labels=glee_result['labels']
    ans={}
    ans['boxes']=merge_boxes
    ans['scores']=merge_scores
    ans['labels']=merge_labels
    res[file_name]=ans


    # merge_result=glee_result
    #
    # return merge_result
    labels = [f"{merge_labels[i]}({merge_scores[i]:.2f})" for i in range(len(merge_scores))]
    integrated_results[file_name] = {
        'boxes': merge_boxes,
        # 'scores': merge_scores,
        'labels': labels,
    }

    image_picture = Image.open(os.path.join(data_path,file_name)).convert("RGB")
    image_with_box=plot_boxes_to_image(image_picture,integrated_results[file_name])

    image_with_box.save(os.path.join(output_dir, file_name))

