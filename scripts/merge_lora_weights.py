import argparse
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
import os
import importlib.util
import sys
# Define the path to the module in the other branch (use absolute path here)
module_path = "/home/malindal/LLaVA-Med/llava/model/builder.py"
# module_path = "/home/malindal/LLaVA-Med/LLaVA/llava/model/builder.py"


# Load the module
spec = importlib.util.spec_from_file_location("builder", module_path)
builder = importlib.util.module_from_spec(spec)
sys.modules["builder"] = builder
spec.loader.exec_module(builder)
load_pretrained_model = builder.load_pretrained_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
