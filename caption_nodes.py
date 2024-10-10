import os
import torch
import subprocess
import folder_paths

from .utils import tensor2pil
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM


# 加载模型
class PhiModelLoder:
    def __init__(self):
        self.loaded_models = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                'model_id': ('STRING', {'default': 'microsoft/Phi-3.5-vision-instruct'})
                , 'attn_implementation': (['flash_attention_2', 'eager'], {'default': 'eager'})
                , 'torch_dtype':(['auto', 'float16', 'bfloat16'], {'default':'auto'})
                , 'trust_remote_code': ('BOOLEAN', {'default': True, })
            },
        }

    RETURN_TYPES = ("MODEL", 'Info')
    RETURN_NAMES = ("model", 'model_info')

    FUNCTION = "ModelLoder"

    OUTPUT_NODE = False

    CATEGORY = "Phi35"

    def ModelLoder(self, model_id, trust_remote_code, torch_dtype, attn_implementation):

        # 判断本地是否有模型文件，如果没有则从huggingface下载模型
        model_path = os.path.join(folder_paths.models_dir, 'LLM', 'Phi-3.5-vision-instruct')
        model_id_list = ['model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']

        for i in model_id_list:
            if not os.path.exists(os.path.join(model_path, i)):
                from huggingface_hub import snapshot_download
                os.makedirs(model_path, exist_ok=True)
                print('开始下载文件，需要魔法')
                # 构建命令
                command = ['huggingface-cli', 'download', '--resume-download', model_id, '--local-dir', model_path]
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print("Download completed:", result.stdout)
                break

        # 判断模型是否已经加载了，如果没有就加载，否则跳过加载
        if model_path in self.loaded_models:
            print('model exists')
            return (self.loaded_models[model_path], self.loaded_models[model_id])

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cuda',
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            _attn_implementation=attn_implementation
        )

        self.loaded_models[model_path] = model
        self.loaded_models[model_id] = model_id

        # 推理模式
        model.eval()

        return (self.loaded_models[model_path], self.loaded_models[model_id])


# 模型推理
class PhiInfer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'model': ('MODEL',)
                , 'model_info': ('Info',)
                , 'image': ('IMAGE',)
                , 'input_prompt': ('STRING', {'default': 'Summarize the deck of slides.', 'multiline': True, })
                , 'max_new_tokens': ('INT', {'default': 512, 'max': 4096, 'min': 512, 'step': 1, 'display': 'number'})
                , 'temperature': ('FLOAT', {'default': 0.5, 'max': 1, 'min': 0, 'step': 0.01, 'display': 'number'})
                , 'do_sample': ('BOOLEAN', {'default': False, })
                , 'skip_special_tokens': ('BOOLEAN', {'default': True, })
                , 'clean_up_tokenization_spaces': ('BOOLEAN', {'default': False, })

            },
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('response',)
    FUNCTION = 'ModelInfer'

    CATEGORY = 'Phi35'

    def ModelInfer(self, model, model_info, image, input_prompt, max_new_tokens, temperature, do_sample,
                   skip_special_tokens, clean_up_tokenization_spaces):

        # 区分单张图片与多张图片，不同图片数量采用不同的num_crops
        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        bs = image.shape[0]

        if bs == 1:
            num_crops = 16
            tensor_list = [image]

            multiline_prompt = input_prompt.splitlines()
            rows = len(multiline_prompt)
            if rows == 1:
                # 单图单轮
                # print('=====单图单轮=====')
                content = f'<|user|>\n<|image_1|>\n{multiline_prompt[0]}<|end|>\n<|assistant|>\n'
                response = self.CoreInfer(model, model_info, num_crops, tensor_list, content, max_new_tokens,
                                          temperature, do_sample, skip_special_tokens, clean_up_tokenization_spaces)
                return (response,)

            else:
                # 单图多轮
                # print('=====单图多轮=====')
                content = f'<|user|>\n<|image_1|>\n{multiline_prompt[0]}<|end|>\n<|assistant|>\n'
                response = ''
                for row in range(rows):
                    response_n = self.CoreInfer(model, model_info, num_crops, tensor_list, content, max_new_tokens,
                                                temperature, do_sample, skip_special_tokens, clean_up_tokenization_spaces)
                    response = response + str(row + 1) + '.' + response_n + '\n'
                    if row + 1 < rows:
                        content += f'{response_n}<|end|>\n<|user|>\n{multiline_prompt[row + 1]}<|end|>\n<|assistant|>\n'
                    else:
                        return (response,)

        else:
            # print('=====多图单轮=====')
            num_crops = 4
            # 将[b,h,w,c]的tensor拆开为一个个[1,h,w,c]的tensor
            tensor_list = list(torch.chunk(image, bs, dim=0))

            placeholder = ""
            for i in range(1, bs + 1):
                placeholder += f"<|image_{i}|>\n"

            content = f'<|user|>\n{placeholder}+{input_prompt}<|end|>\n<|assistant|>\n'
            response = self.CoreInfer(model, model_info, num_crops, tensor_list, content, max_new_tokens, temperature,
                                      do_sample, skip_special_tokens, clean_up_tokenization_spaces)
            return (response,)

    def CoreInfer(self, model, model_info, num_crops, tensor_list, content, max_new_tokens, temperature, do_sample,
                  skip_special_tokens, clean_up_tokenization_spaces):
        processor = AutoProcessor.from_pretrained(model_info, trust_remote_code=True, num_crops=num_crops)
        image_list = [tensor2pil(t) for t in tensor_list]

        messages = [
            {"role": "user", "content": content},
        ]

        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # print(f'====={prompt}=====')

        inputs = processor(prompt, image_list, return_tensors="pt").to("cuda")

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
        }

        with torch.no_grad():
            generate_ids = model.generate(**inputs,
                                          eos_token_id=processor.tokenizer.eos_token_id,
                                          **generation_args
                                          )

        # remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids,
                                          skip_special_tokens=skip_special_tokens,
                                          clean_up_tokenization_spaces=clean_up_tokenization_spaces)[0]

        torch.cuda.empty_cache()
        # print(f'====={response}=====')

        return response


NODE_CLASS_MAPPINGS = {
     "PhiModelLoder": PhiModelLoder
    ,'PhiInfer': PhiInfer
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
     "PhiModelLoder": "PhiModelLoder"
    ,'PhiInfer': 'PhiInfer'
}
