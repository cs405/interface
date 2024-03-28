# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import json
import io
import zipfile

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from demo.predictor import VisualizationDemo
from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"


class Result:
    def __init__(self, id, image, value):
        self.id = id
        self.image = image
        self.value = value


def result_encoder(obj):
    if isinstance(obj, Result):
        return {'id': obj.id, 'image': obj.image, 'deepsolo': obj.value}
    return json.JSONEncoder.default(obj)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        dest='config_file',
        # default='/data1/xyj/DeepSolo-main/configs/ViTAEv2_S/TotalText/finetune_150k_tt_mlt_13_15_textocr.yaml',
        # metavar="FILE",
        help="path to config file",
    )
    # '-f',
    # '--config-file',
    # dest='config_file',
    # type = argparse.FileType(mode='r'),
    # default=yaml_path
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+",
                        help="A list of space separated input images")
    parser.add_argument(
        "--output",
        default="/data1/xyj/DeepSolo-main/output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS',
                 '/data1/xyj/DeepSolo-main/work_dirs/tt_vitaev2-s_finetune_synth-tt-mlt-13-15-textocr.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser

from flask import Flask, request

app = Flask(__name__)


@app.route('/OCR_deepsolo_en', methods=['GET', 'POST'])
def OCR_deepsolo_en():
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

    # 选择进程的启动方法。
    mp.set_start_method("spawn", force=True)
    config_file="./configs/ViTAEv2_S/TotalText/finetune_150k_tt_mlt_13_15_textocr.yaml"
    output="./output/en"
    # ckpt=[]
    args = get_parser().parse_args(["--input", image_folder,
                                    "--config-file",config_file,
                                    "--output",output,
                                     "--opts", "MODEL.WEIGHTS","./work_dirs/tt_vitaev2-s_finetune_synth-tt-mlt-13-15-textocr.pth"])

    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        output_path = os.path.join(args.output, "deepsolo_en.json")
        if (os.path.exists(output_path)):
            os.remove(output_path)
        if os.path.exists(image_folder):
            print(f"{image_folder} path is not exist")
            ans = {}
            for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                start_time = time.time()
                predictions, visualized_output, text_output = demo.run_on_image(img)
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                    res = Result(os.path.basename(path), out_filename, text_output)

                    with open(output_path, "a", encoding="utf8") as file:
                        json.dump(result_encoder(res), file, ensure_ascii=False, indent=4)
                    # predictions.save(out_filename)
                    ans[path] = text_output

                else:
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
            # 将列表转换为 JSON 格式的字符串
            json_data = json.dumps(ans, ensure_ascii=False)

            # 将 JSON 字符串写入文件
            with open("data.json", "w") as file:
                file.write(json_data)
            return {'result': json_data}
        else:
            return {'error': 'the file path of incoming has a problem,please check it'}
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

@app.route('/OCR_deepsolo_zh_ViTAEv2_S', methods=['GET', 'POST'])
def OCR_deepsolo_zh1():
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


    # image_folder = request.json['image_folder']
    # 选择进程的启动方法。
    mp.set_start_method("spawn", force=True)
    config_file="./configs/ViTAEv2_S/ReCTS/finetune.yaml"
    output="./output/zh_ViTAEv2_S"
    # ckpt=()
    args = get_parser().parse_args(["--input", image_folder,
                                    "--config-file",config_file,
                                    "--output",output,
                                    "--opts","MODEL.WEIGHTS","./work_dirs/rects_vitaev2-s_finetune.pth"])
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        output_path = os.path.join(args.output, "deepsolo_zh_ViTAEv2_S.json")
        if (os.path.exists(output_path)):
            os.remove(output_path)
        if os.path.exists(image_folder):
            print(f"{image_folder} path is not exist")
            ans = {}
            for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                start_time = time.time()
                predictions, visualized_output, text_output = demo.run_on_image(img)
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                    res = Result(os.path.basename(path), out_filename, text_output)

                    with open(output_path, "a", encoding="utf8") as file:
                        json.dump(result_encoder(res), file, ensure_ascii=False, indent=4)
                    # predictions.save(out_filename)
                    ans[path] = text_output

                else:
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
            # 将列表转换为 JSON 格式的字符串
            json_data = json.dumps(ans, ensure_ascii=False)

            # 将 JSON 字符串写入文件
            with open("data.json", "w") as file:
                file.write(json_data)
            return {'result': json_data}
        else:
            return {'error': 'the file path of incoming has a problem,please check it'}
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

@app.route('/OCR_deepsolo_zh_R50', methods=['GET', 'POST'])
def OCR_deepsolo_zh2():
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


    # image_folder = request.json['image_folder']
    # 选择进程的启动方法。
    mp.set_start_method("spawn", force=True)
    config_file="./configs/R_50/ReCTS/finetune.yaml"
    output="./output/zh_R50"
    # ckpt=()
    args = get_parser().parse_args(["--input", image_folder,
                                    "--config-file",config_file,
                                    "--output",output,
                                    "--opts","MODEL.WEIGHTS","./work_dirs/rects_res50_finetune.pth"])
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        output_path = os.path.join(args.output, "deepsolo_zh_R50.json")
        if (os.path.exists(output_path)):
            os.remove(output_path)
        if os.path.exists(image_folder):
            print(f"{image_folder} path is not exist")
            ans = {}
            for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                start_time = time.time()
                predictions, visualized_output, text_output = demo.run_on_image(img)
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                    res = Result(os.path.basename(path), out_filename, text_output)

                    with open(output_path, "a", encoding="utf8") as file:
                        json.dump(result_encoder(res), file, ensure_ascii=False, indent=4)
                    # predictions.save(out_filename)
                    ans[path] = text_output

                else:
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
            # 将列表转换为 JSON 格式的字符串
            json_data = json.dumps(ans, ensure_ascii=False)

            # 将 JSON 字符串写入文件
            with open("data.json", "w") as file:
                file.write(json_data)
            return {'result': json_data}
        else:
            return {'error': 'the file path of incoming has a problem,please check it'}
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5021)
