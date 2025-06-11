# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import json
import cv2
import numpy as np
from typing import Tuple
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from einops import rearrange
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from detectron2.data import MetadataCatalog
from skysense_o import SkySenseO
from PIL import Image
import torchvision.transforms as transforms
from colorama import init, Fore, Style
from visualizer import visual_ann
import pyfiglet


@META_ARCH_REGISTRY.register()
class SkySenseO_DEMO(SkySenseO):

    def __init__(self, cfg):

        super().__init__(cfg)
        # Config
        init(autoreset=True)
        self.custom_image = False
        self.custom_text = True
        self.custom_save_path = False
        self.retrieval_augmentation = False
        # Default Parameters
        self.default_image_path = "/gruntdata/rs_nas/workspace/wenshuo.ljw/project/CATSEG/datasets/tent_seg/ref_image_crop_2.png"
        self.default_text = "open_world"  # others,structrue
        self.default_save_path = "./demo_result.png"

    def format_text(self, input_text):
        """
        this text input could be refactor three formats:
        1. open-world interpretation.  input format: open_world
        2. denote specific dataset classes.  input format: xxx_dataset
        3. random input text  (1) with backgroud retrieval augmentation (2) with background threshold. input format: random text (TODO)
        """
        # 1) open-world interpretation.
        if input_text == "open_world":
            test_class_json = "/gruntdata/rs_nas/workspace/wenshuo.ljw/project/CATSEG/datasets/sky5k/sky5k.json"
            with open(test_class_json, 'r') as f_in:
                test_class_texts = json.load(f_in)
        # 2) denote specific dataset classes.         
        elif isinstance(input_text, str) and input_text.endswith("_dataset"):
            dataset_name = input_text.split("_")[0]
            test_class_json = f"datasets/{dataset_name}/{dataset_name}.json"
            with open(test_class_json, 'r') as f_in:
                test_class_texts = json.load(f_in)
        # 3) random input text 
        elif ',' in input_text:
            test_class_texts = input_text.split(',')
        # 4) interaction setting  
        elif input_text == "setting": # T or F
            custom_image_setting = input("Custom_image: ") 
            custom_text_setting = input("Custom_text: ")
            custom_save_path_setting = input("Custom_save_path: ")
            self.custom_image = True if custom_image_setting == "T" else False
            self.custom_text = True if custom_text_setting == "T" else False
            self.custom_save_path = True if custom_save_path_setting == "T" else False
            test_class_texts = self.default_text
        else:
            print(Fore.RED + Style.BRIGHT + f"Your input text is unvalid, the default text {self.default_text} will be used.")
            test_class_texts = self.default_text

        if self.retrieval_augmentation:
            retrived_texts = {}
            retrieval_nums = {}
            retrieved_result_path = f"datasets/{dataset_name}/o1.json"
            with open(retrieved_result_path, 'r') as rap:
                retrieved_result = json.load(rap)
                retrieval_num = [len(open_add_values) for open_add_keys, open_add_values in retrieved_result.items() if open_add_keys != "except"]  
                if "except" in retrieved_result.keys():
                    retrieved_result.pop("except")           
                retrived_text = [item for sublist in retrieved_result.values() for item in sublist]
            test_class_texts = test_class_texts + retrived_text
            return test_class_texts, retrieval_num 
        else:
            return test_class_texts
            
    def generate_input_format(self, image_path, input_text, save_path):

        image = Image.open(image_path)
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image)
        image_tensor = (image_tensor * 255).to(torch.uint8)
        if self.retrieval_augmentation:
            test_class_texts, retrieval_nums, retrived_texts = self.format_text(input_text)
            output = {
                'file_name': image_path,
                'text': test_class_texts,
                'width': image.width,
                'height': image.height,
                'image': image_tensor,
                'retrieval_nums': retrieval_nums,
                'retrived_texts': retrived_texts,
                'save_path': save_path
            }
        else:
            test_class_texts = self.format_text(input_text)
            output = {
                'file_name': image_path,
                'text': test_class_texts,
                'width': image.width,
                'height': image.height,
                'image': image_tensor,
                'save_path': save_path
            }
        return output

    def entrance_interaction(self):
        # You can close some specific options for efficient interaction.
        input() if (not self.custom_image) and (not self.custom_text) and (not self.custom_save_path) else None
        image_path = None
        input_text = None
        save_path = None
        if self.custom_image:
            print(Fore.BLUE + Style.BRIGHT + "Please input your input image path: ", end="")
            image_path = input()
        if (not image_path) or (not self.custom_image):
            image_path = self.default_image_path

        if self.custom_text:
            print(Fore.MAGENTA + Style.BRIGHT + "Please input your target texts with ',' split: ", end="")
            input_text = input()
        if (not input_text) or (not self.custom_text):
            input_text = self.default_text
            
        if self.custom_save_path:
            print(Fore.GREEN + Style.BRIGHT + "Please input your save path: ", end="")
            save_path = input()
        if (not save_path) or (not self.custom_save_path):
            save_path = self.default_save_path
        
        return image_path, input_text, save_path

    @torch.no_grad()
    def demo(self):         
        # Entrance
        image_path, input_text, save_path = self.entrance_interaction()
        print(Fore.RED + Style.BRIGHT + "Running...")
        batched_inputs = self.generate_input_format(image_path, input_text, save_path)
        # Image Preprocessing
        images = batched_inputs["image"].to(self.device)
        clip_images = [(images - self.clip_pixel_mean) / self.clip_pixel_std]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
        # Visual Features from SkySense-O backbone
        clip_features = self.sem_seg_head.clip_model.encode_image(clip_images_resized, dense=True)
        image_feats = clip_features[-1]
        res2 = self.visual_guidance_conv1(clip_features[-3])
        res1 = self.visual_guidance_conv2(clip_features[-4])  
        visual_guidance = [image_feats, res2, res1]
        # Text Features from SkySense-O backbone
        self.test_class_text = batched_inputs["text"]
        text_feats = self.sem_seg_head.get_text_embeds(self.test_class_text, self.prompt_templates, self.sem_seg_head.clip_model)
        text_feats = text_feats.repeat(image_feats.shape[0], 1, 1, 1)
        # Visual Decoder
        outputs = self.sem_seg_head(image_feats, text_feats, visual_guidance)
        #  Inference
        outputs = outputs.sigmoid()
        test_classes_num = len(self.test_class_text)
        if self.retrieval_augmentation:
            edge_num_counter = 0 
            self.retrieval_num = batched_inputs["retrieval_nums"]
            for node_index, edge_num in enumerate(self.retrieval_num):
                if edge_num and node_index!=(len(self.retrieval_num)-1):
                    outputs[0,node_index,:,:] = torch.max(torch.cat((outputs[0, node_index].unsqueeze(0), 
                                                                        outputs[0, (test_classes_num + edge_num_counter):(test_classes_num + edge_num_counter + edge_num)])),
                                                                        dim=0)[0]
                    edge_num_counter = edge_num_counter + edge_num
                elif edge_num and node_index==(len(self.retrieval_num)-1):
                    outputs[0,node_index,:,:] = torch.max(torch.cat((outputs[0, node_index].unsqueeze(0), 
                                                                        outputs[0, (test_classes_num + edge_num_counter):])), dim=0)[0]
        outputs = outputs[:,:test_classes_num,:,:]
        output_height = batched_inputs["height"]
        output_width = batched_inputs["width"]
        output = sem_seg_postprocess(outputs[0], clip_images.image_sizes[0], output_height, output_width)
        output = torch.argmax(output, dim=0)

        visual_ann(output.cpu().numpy(), batched_inputs["file_name"], batched_inputs["save_path"], self.test_class_text)
        # cv2.imwrite(batched_inputs["save_path"], output.cpu().numpy())
        print(Fore.YELLOW + Style.BRIGHT + f"Finished! The result is saved in {batched_inputs['save_path']}")

    def run_demo(self):

        print(pyfiglet.figlet_format("SkySense - O"))
        while True:
            self.demo()