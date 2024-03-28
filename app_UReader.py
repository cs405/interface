import itertools
import os
import traceback
import json
import torch
from pathlib import Path
from pipeline.utils import add_config_args, set_args
import argparse
from sconf import Config
from mplug_owl.processing_mplug_owl import MplugOwlProcessor
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.configuration_mplug_owl import MplugOwlConfig
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
import torch
from pipeline.data_utils.processors.builder import build_processors
from pipeline.data_utils.processors import *
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from PIL import Image

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
        return {'id': obj.id,  'prompt':obj.prompt, 'UReader': obj.value}
    return json.JSONEncoder.default(obj)

class InferenceDataset(torch.utils.data.Dataset):

    def __init__(self, jsonl):
        with open(jsonl, 'r', encoding="utf-8") as f:
            self.lines = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


def collate_fn(batches):
    return batches


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

from flask import Flask, request

app = Flask(__name__)

@app.route('/OCR_UReader_en', methods=['GET','POST'])
def OCR_UReader_en():
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_checkpoint', type=str, default=None,
                        help='Path to the trained checkpoint. If given, evaluate the given weights instead of the one in hf model.')
    parser.add_argument('--hf_model', type=str, default='./ckpt/ureader-v1',
                        help='Path to the huggingface model')
    args = parser.parse_args()
    config = Config('configs/sft/release.yaml')
    add_config_args(config, args)
    set_args(args)
    if not os.environ.get('MASTER_ADDR', None):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '54141'
        os.environ['LOCAL_RANK'] = '0'
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    image_processor = build_processors(config['valid_processors'])['sft']

    if args.eval_checkpoint:
        state_dict = torch.load(args.eval_checkpoint)
    else:
        state_dict = None

    tokenizer = LlamaTokenizer.from_pretrained(args.hf_model)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    model = MplugOwlForConditionalGeneration.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float,
        state_dict=state_dict,
    )

    model.half()
    model.cuda()
    model.eval()

    # image_folder = "/data1/xyj/datasets/en_test"
    output_path = 'outputs/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, "en_test_UReader.json")
    if (os.path.exists(output_path)):
        os.remove(output_path)

    question = 'recognize all the texts in the image.'
    prompt = f'Human: <image>\nHuman: {question}\nAI: '

    if os.path.exists(image_folder):
        print(f"{image_folder} path is not exist")
        ans = {}
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            images = Image.open(image_path).convert('RGB')

            inputs = processor(text=prompt, images=images, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items()}
            try:
                res = model.generate(**inputs, top_p=1, repetition_penalty=1.15, max_new_tokens=1024, do_sample=False)
                model_answer = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
            except Exception as e:
                model_answer = ''
                print(traceback.format_exc())

            # print(question)
            print('model:', model_answer)
            # print('\n')
            res = Result(filename, question, model_answer)
            with open(output_path, "a", encoding="utf8") as file:
                json.dump(result_encoder(res), file, ensure_ascii=False, indent=4)
            ans[image_path] = model_answer
        # 将列表转换为 JSON 格式的字符串
        json_data = json.dumps(ans, ensure_ascii=False)

        # 将 JSON 字符串写入文件
        with open("data.json", "w") as file:
            file.write(json_data)
        return {'result': json_data}
    else:
        return {'error':'the file path of incoming has a problem,please check it'}
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5025)


