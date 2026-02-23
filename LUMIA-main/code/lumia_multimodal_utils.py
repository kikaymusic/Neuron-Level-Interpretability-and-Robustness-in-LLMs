import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXModel
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np
import torch
import torch.nn.functional as F

import bitsandbytes as bnb 
from tensorflow.keras import layers, models, initializers, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from baukit import Trace, TraceDict
from datasets import load_dataset, Dataset
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def extract_samples_textcap(
    member_samples, not_member_sample, samples_analyze, model, processor
):
    members_visual = []
    members_lang = []


    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "<image>\nProvide a one-sentence caption for the provided image.".replace(
                            "<image>", ""
                        ),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, member_samples[i]["images"], 0
        )
        members_visual.append(visual_part)
        members_lang.append(language_part)

    members_visual = np.array(members_visual)
    members_lang = np.array(members_lang)
    nonmembers_visual = []
    nonmembers_lang = []


    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "<image>\nProvide a one-sentence caption for the provided image.".replace(
                            "<image>", ""
                        ),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, not_member_sample[i]["images"], 0
        )
        nonmembers_visual.append(visual_part)
        nonmembers_lang.append(language_part)

    nonmembers_visual = np.array(nonmembers_visual)
    nonmembers_lang = np.array(nonmembers_lang)
    return members_visual, members_lang, nonmembers_visual, nonmembers_lang


def extract_samples_math(
    member_samples, not_member_sample, samples_analyze, model, processor
):
    members_visual = []
    members_lang = []


    i = 0
    while len(members_visual) < 500:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": member_samples[i]["conversations"][0]["value"]
                        .replace("\n", " ")
                        .replace("<image>", ""),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, member_samples[i]["images"], 0
        )

        if len(visual_part) != 0 and len(language_part) != 0:
            
            members_visual.append(visual_part)
            members_lang.append(language_part)
        i += 1
    members_visual = np.array(members_visual)
    members_lang = np.array(members_lang)
    nonmembers_visual = []
    nonmembers_lang = []

    i = 0
    while len(nonmembers_visual) < 500:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": not_member_sample[i]["conversations"][0][
                            "value"
                        ].replace("<image>", ""),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, not_member_sample[i]["images"], 0
        )
        if len(visual_part) != 0 and len(language_part) != 0:
            
            nonmembers_visual.append(visual_part)
            nonmembers_lang.append(language_part)
        i += 1

    nonmembers_visual = np.array(nonmembers_visual)
    nonmembers_lang = np.array(nonmembers_lang)
    return members_visual, members_lang, nonmembers_visual, nonmembers_lang


def extract_samples_aok(
    member_samples, not_member_sample, samples_analyze, model, processor
):
    members_visual = []
    members_lang = []


    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": member_samples[i]["conversations"][0]["value"].replace(
                            "<image>", ""
                        ),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, member_samples[i]["image"], 0
        )
        members_visual.append(visual_part)
        members_lang.append(language_part)

    members_visual = np.array(members_visual)
    members_lang = np.array(members_lang)
    nonmembers_visual = []
    nonmembers_lang = []

    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": not_member_sample[i]["conversations"][0][
                            "value"
                        ].replace("<image>", ""),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, not_member_sample[i]["image"], 0
        )
        nonmembers_visual.append(visual_part)
        nonmembers_lang.append(language_part)

    nonmembers_visual = np.array(nonmembers_visual)
    nonmembers_lang = np.array(nonmembers_lang)
    return members_visual, members_lang, nonmembers_visual, nonmembers_lang


def extract_samples_science(
    member_samples, not_member_sample, samples_analyze, model, processor
):
    members_visual = []
    members_lang = []


    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": member_samples[i]["conversations"][0]["value"].replace(
                            "<image>", ""
                        ),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, member_samples[i]["image"], 0
        )
        members_visual.append(visual_part)
        members_lang.append(language_part)

    members_visual = np.array(members_visual)
    members_lang = np.array(members_lang)
    nonmembers_visual = []
    nonmembers_lang = []


    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": not_member_sample[i]["conversations"][0][
                            "value"
                        ].replace("<image>", ""),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, not_member_sample[i]["image"], 0
        )
        nonmembers_visual.append(visual_part)
        nonmembers_lang.append(language_part)

    nonmembers_visual = np.array(nonmembers_visual)
    nonmembers_lang = np.array(nonmembers_lang)
    return members_visual, members_lang, nonmembers_visual, nonmembers_lang


def load_multimodal(model_name):
    model_id = model_name
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        output_hidden_states=True,
        return_dict=True,
        torch_dtype=torch.float16,
    ).to("cuda:0")


    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def process_members_not_members_icon(seed):
    dataset_name = "lmms-lab/ICON-QA"
    ds_math = load_dataset(dataset_name)
    member_dataset = load_dataset("lmms-lab/LLaVA-OneVision-Data", "IconQA(MathV360K)")
    images, conversations = [], []
    for i in enumerate(ds_math["val"]):
        images.append(i[1]["query_image"])
        conversations.append(i[1]["question"])

    data_dict = {"image": images, "conversations": conversations}

    non_member_dataset = Dataset.from_dict(data_dict)

    non_member_sample = non_member_dataset.shuffle(seed=seed).select(range(1000))
    member_sample = member_dataset["train"].shuffle(seed=seed).select(range(1000))

    return non_member_sample, member_sample


def extract_samples_icon(
    member_samples, not_member_sample, samples_analyze, model, processor
):
    members_visual = []
    members_lang = []


    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f'<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: {member_samples[i]["conversations"]}'.replace(
                            "<image>", ""
                        ),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, member_samples[i]["image"], 0
        )
        members_visual.append(visual_part)
        members_lang.append(language_part)

    members_visual = np.array(members_visual)
    members_lang = np.array(members_lang)
    nonmembers_visual = []
    nonmembers_lang = []

    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": not_member_sample[i]["conversations"][0][
                            "value"
                        ].replace("<image>", ""),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, not_member_sample[i]["image"], 0
        )
        nonmembers_visual.append(visual_part)
        nonmembers_lang.append(language_part)

    nonmembers_visual = np.array(nonmembers_visual)
    nonmembers_lang = np.array(nonmembers_lang)

    return members_visual, members_lang, nonmembers_visual, nonmembers_lang


def process_members_not_members_science(seed):
    ds_math = load_dataset("derek-thomas/ScienceQA")
    member_dataset = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data", "scienceqa(cauldron,llava_format)"
    )


    images, conversations = [], []

    for i in enumerate(ds_math["validation"]):
        images.append(i[1]["image"])

        question = i[1]["question"]
        choices = i[1]["choices"]
        hint = i[1]["hint"]
        final_question = f"{question}\nChoices:"
        for choice in choices:
            final_question = f"{final_question}\n{choice}"
        final_question = f"{final_question}\nHint:{hint}"
        sol = i[1]["solution"]

        conversation = [
            {"from": "human", "value": f"{final_question}"},
            {"from": "gpt", "value": f"{sol}"},
        ]
        conversations.append(conversation)

    data_dict = {"image": images, "conversations": conversations}

    non_member_dataset = Dataset.from_dict(data_dict)

    non_member_sample = non_member_dataset.shuffle(seed=seed).select(range(1000))
    member_sample = member_dataset["train"].shuffle(seed=seed).select(range(1000))


    return non_member_sample, member_sample


def extract_samples_clevr(
    member_samples, not_member_sample, samples_analyze, model, processor
):
    members_visual = []
    members_lang = []


    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f'\nHint: Please answer the question and provide the final answer at the end.\nQuestion: {member_samples[i]["conversations"]}'.replace(
                            "<image>", ""
                        ),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, member_samples[i]["images"], 0
        )
        members_visual.append(visual_part)
        members_lang.append(language_part)

    members_visual = np.array(members_visual)
    members_lang = np.array(members_lang)
    nonmembers_visual = []
    nonmembers_lang = []

    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": not_member_sample[i]["conversations"][0][
                            "value"
                        ].replace("<image>", ""),
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, not_member_sample[i]["image"], 0
        )
        nonmembers_visual.append(visual_part)
        nonmembers_lang.append(language_part)

    nonmembers_visual = np.array(nonmembers_visual)
    nonmembers_lang = np.array(nonmembers_lang)

    return members_visual, members_lang, nonmembers_visual, nonmembers_lang


def process_members_not_members_aok(seed):
    ds_math = load_dataset("HuggingFaceM4/A-OKVQA")
    member_dataset = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data", "aokvqa(cauldron,llava_format)"
    )

    images, conversations = [], []
    for i in enumerate(ds_math["validation"]):
        qr = i[1]["question"]
        images.append(i[1]["image"])
        conversations.append(
            [
                {
                    "from": "human",
                    "value": f"{qr}\nAnswer the question using a single word or phrase.",
                },
                {"from": "gpt", "value": "Tails."},
            ]
        )
    data_dict = {"image": images, "conversations": conversations}

    non_member_dataset = Dataset.from_dict(data_dict)

    non_member_sample = non_member_dataset.shuffle(seed=seed).select(range(1000))
    member_sample = member_dataset["train"].shuffle(seed=seed).select(range(1000))


    return non_member_sample, member_sample


def process_members_not_members_aok2(seed):
    ds_math = load_dataset("lmms-lab/OK-VQA")
    member_dataset = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data", "aokvqa(cauldron,llava_format)"
    )

    images, conversations = [], []
    for i in enumerate(ds_math["val2014"]):
        qr = i[1]["question"]
        images.append(i[1]["image"])
        conversations.append(
            [
                {
                    "from": "human",
                    "value": f"{qr}\nAnswer the question using a single word or phrase.",
                },
                {"from": "gpt", "value": "Tails."},
            ]
        )
    data_dict = {"image": images, "conversations": conversations}

    non_member_dataset = Dataset.from_dict(data_dict)

    non_member_sample = non_member_dataset.shuffle(seed=seed).select(range(1000))
    member_sample = member_dataset["train"].shuffle(seed=seed).select(range(1000))


    return non_member_sample, member_sample


def process_members_not_members_chartqa(seed):
    dataset_name = "lmms-lab/ChartQA"
    ds = load_dataset(dataset_name)
    member_dataset = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data", "chartqa(cauldron,llava_format)"
    )

    images, conversations = [], []
    for i in enumerate(ds["test"]):
        qr = i[1]["question"]
        images.append(i[1]["image"])
        conversations.append(qr)
    data_dict = {"image": images, "conversations": conversations}

    non_member_dataset = Dataset.from_dict(data_dict)

    non_member_sample = non_member_dataset.shuffle(seed=seed).select(range(1000))
    member_sample = member_dataset["train"].shuffle(seed=seed).select(range(1000))



    return non_member_sample, member_sample


def extract_samples_chart(
    member_samples, not_member_sample, samples_analyze, model, processor
):
    members_visual = []
    members_lang = []

    for i in range(samples_analyze):
        qr = (
            member_samples[i]["conversations"]
            + "\nAnswer the question using a single word or phrase."
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": qr},
                ],
            },
        ]  

        width, height = member_samples[i]["image"].size
        if height < 1000:

            visual_part, language_part = get_activations_baukit(
                model, processor, conversation, member_samples[i]["image"], 0
            )
            members_visual.append(visual_part)
            members_lang.append(language_part)

    members_visual = np.array(members_visual)
    members_lang = np.array(members_lang)
    nonmembers_visual = []
    nonmembers_lang = []


    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": not_member_sample[i]["conversations"][0][
                            "value"
                        ].replace("<image>", ""),
                    },
                ],
            },
        ]
        width, height = not_member_sample[i]["image"].size
        if height < 1000:

            visual_part, language_part = get_activations_baukit(
                model, processor, conversation, not_member_sample[i]["image"], 0
            )
            nonmembers_visual.append(visual_part)
            nonmembers_lang.append(language_part)

    nonmembers_visual = np.array(nonmembers_visual)
    nonmembers_lang = np.array(nonmembers_lang)
    return members_visual, members_lang, nonmembers_visual, nonmembers_lang


def extract_samples_magpie(samples_analyze, model, processor):
    dataset_name = "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1"
    # Load datasets
    ds = load_dataset(dataset_name)
    ds2 = load_dataset("lmms-lab/LLaVA-OneVision-Data", "magpie_pro(l3_80b_mt)")
    ds = ds["train"].shuffle(seed=3).select(range(50000))

    # Extract UUIDs
    ids = ds["uuid"]
    uuids = ds2["train"]["id"]

    # Convert lists to numpy arrays for faster comparison
    ids_array = np.array(ids)
    uuids_array = np.array(uuids)

    # Find common elements (members)
    members = np.intersect1d(ids_array, uuids_array)

    # Find non-members (elements in ids but not in uuids)
    non_members = np.setdiff1d(ids_array, uuids_array)



    # Filter the dataset to create members and non-members datasets
    members_mask = np.isin(ids_array, members)
    non_members_mask = np.isin(ids_array, non_members)

    # Create new datasets based on the mask
    members_ds = ds.filter(lambda example: example["uuid"] in members)
    non_members_ds = ds.filter(lambda example: example["uuid"] in non_members)

    members_visual = []
    members_lang = []


    for i in range(samples_analyze):

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": members_ds["conversations"][i][0]["value"],
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, "none", 0
        )
        members_visual.append(visual_part)
        members_lang.append(language_part)

    members_visual = np.array(members_visual)
    members_lang = np.array(members_lang)
    nonmembers_visual = []
    nonmembers_lang = []


    for i in range(samples_analyze):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": members_ds["conversations"][i][0]["value"],
                    },
                ],
            },
        ]

        visual_part, language_part = get_activations_baukit(
            model, processor, conversation, "none", 0
        )
        nonmembers_visual.append(visual_part)
        nonmembers_lang.append(language_part)

    nonmembers_visual = np.array(nonmembers_visual)
    nonmembers_lang = np.array(nonmembers_lang)
    return members_visual, members_lang, nonmembers_visual, nonmembers_lang


def process_members_not_members(seed):
    ds_math = load_dataset("Zhiqiang007/MathV360K")
    ids_math = set(ds_math["train"]["id"])
    dictionary_members = {}
    for ids in ids_math:
        dictionary_members[ids] = 1
    member_dataset = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data", "CLEVR-Math(MathV360K)"
    )
    ids_clevr = member_dataset["train"]["id"]
    for ids in ids_clevr:
        if ids in dictionary_members:
            dictionary_members[ids] = 0

    non_members = []
    for ids, value in enumerate(dictionary_members):
        if value:
            non_members.append(ids)

    non_members = random.sample(non_members, len(ids_clevr))
    dict_from_list = {non_members[i]: i for i in range(len(non_members))}
    images, ids, conversations = [], [], []
    imagesM, idsM, conversationsM = [], [], []
    for i in enumerate(ds_math["train"]):
        if i[0] in dict_from_list:
            images.append(
                f'/home/oso/Documents/mia/mia_paper/data_images/data_images/{i[1]["image"]}'
            )
            ids.append(i[1]["id"])
            conversations.append(i[1]["conversations"])
        else:
            imagesM.append(
                f'/home/oso/Documents/mia/mia_paper/data_images/data_images/{i[1]["image"]}'
            )
            idsM.append(i[1]["id"])
            conversationsM.append(i[1]["conversations"])

    data_dict = {"images": images, "ids": ids, "conversations": conversations}
    data_dictM = {"images": imagesM, "ids": idsM, "conversations": conversationsM}
    non_member_dataset = Dataset.from_dict(data_dict)
    member_dataset = Dataset.from_dict(data_dictM)

    non_member_sample = non_member_dataset.shuffle(seed=seed).select(range(1000))
    member_sample = member_dataset.shuffle(seed=seed).select(range(1000))

    return non_member_sample, member_sample


def process_members_not_members_textcaps(seed):
    ds_math = load_dataset("lmms-lab/TextCaps")
    images, ids, conversations = [], [], []
    imagesM, idsM, conversationsM = [], [], []
    count = 0
    for i in enumerate(ds_math["val"]):
        imagesM.append(i[1]["image"])
        conversationsM.append(i[1]["question"])

    count = 0
    for i in enumerate(ds_math["train"]):
        images.append(i[1]["image"])
        conversations.append(i[1]["question"])


    data_dict = {"images": images, "conversations": conversations}
    data_dictM = {"images": imagesM, "conversations": conversationsM}
    non_member_dataset = Dataset.from_dict(data_dict)
    member_dataset = Dataset.from_dict(data_dictM)

    non_member_sample = non_member_dataset.shuffle(seed=seed).select(range(3000))
    member_sample = member_dataset.shuffle(seed=seed).select(range(3000))

    return non_member_sample, member_sample



def get_activations_baukit(model, processor, conversation, image, not_member):
    modules = []

    hook_layers_l = [
        f"language_model.model.layers.{l}.mlp"
        for l in range(len(model.language_model.model.layers))
    ]
    hook_layers = [
        f"vision_tower.vision_model.encoder.layers.{l}.mlp"
        for l in range(len(model.vision_tower.vision_model.encoder.layers))
    ]
    hook_layers = hook_layers + hook_layers_l
    # hook_layers = [f'vision_tower.vision_model.encoder.layers.{l}' for l in range(24)]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    if not_member:
        raw_image = image
    else:
        if type(image) == str:
            if "none" in image:
                raw_image = Image.new("RGB", (5, 5))
            else:
                raw_image = Image.open(image)

        else:
            if image is None:
                raw_image = Image.new("RGB", (5,5)) 
            else:
                raw_image = image
        # raw_image = raw_image.resize((384,384))


    width, height = raw_image.size

    if width < 1000 and height < 1000:
        with torch.no_grad():
            inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(
                "cuda:0", torch.float16
            )
            del raw_image
            del prompt
            torch.cuda.empty_cache()
            with TraceDict(
                model, layers=hook_layers, retain_input=True, retain_output=True
            ) as rep:
                outputs = model.generate(
                    **inputs,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1,
                    min_new_tokens=1,
                    return_dict=True,
                )

                visual_part = []
                language_part = []
                for i in range(len(hook_layers)):
                    if "vision" in hook_layers[i]:
                        if hasattr(rep[hook_layers[i]], "output"):
                            visual_representation = (
                                rep[hook_layers[i]].output[0].detach().cpu().numpy()
                            )

                            visual_representation = np.expand_dims(
                                visual_representation, axis=0
                            )

                            mean = visual_representation.mean(axis=1)

                            visual_part.append(mean)
                        else:
                            zero_array = np.zeros((1, 1152))
                            visual_part.append(zero_array)

                    else:
                        language_representation = (
                            rep[hook_layers[i]].output[0].detach().cpu().numpy()
                        )

                        language_representation = np.expand_dims(
                            language_representation, axis=0
                        )

                        mean = language_representation.mean(axis=1)

                        language_part.append(mean)
            visual_part = np.array(visual_part)
            language_part = np.array(language_part)
    else: 
        visual_part = []
        language_part = []

        
    return visual_part, language_part


class Classifier(tf.keras.Model):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()

        self.fc1 = layers.Dense(512, kernel_initializer=initializers.HeUniform(), input_dim=input_dim)
        self.fc2 = layers.Dense(256, kernel_initializer=initializers.HeUniform())
        self.fc3 = layers.Dense(128, kernel_initializer=initializers.HeUniform())
        self.fc4 = layers.Dense(64, kernel_initializer=initializers.HeUniform())
        self.fc5 = layers.Dense(32, kernel_initializer=initializers.HeUniform())
        self.fc6 = layers.Dense(1)  # Output layer for binary classification
        self.dropout = layers.Dropout(0.5)  # 50% dropout rate
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()
        self.bn5 = layers.BatchNormalization()
    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.bn1(x)
        x =tf.keras.activations.elu(
        x, alpha=1.0
            )
        x = self.dropout(x, training=training)

        x = self.fc2(x)
        x = self.bn2(x)
        x =tf.keras.activations.elu(
        x, alpha=1.0
            )
        x = self.dropout(x, training=training)

        x = self.fc3(x)
        x = self.bn3(x)
        x =tf.keras.activations.elu(
        x, alpha=1.0
            )
        x = self.dropout(x, training=training)

        x = self.fc4(x)
        x = self.bn4(x)
        x =tf.keras.activations.elu(
        x, alpha=1.0
            )
        x = self.dropout(x, training=training)

        x = self.fc5(x)
        x = self.bn5(x)
        x =tf.keras.activations.elu(
        x, alpha=1.0
            )
        x = self.dropout(x, training=training)

        x = self.fc6(x)
        x = tf.sigmoid(x)
        return x


def build_classifier(input_dim):
    classifier = Classifier(input_dim)

    # Define the loss function
    criterion = losses.BinaryCrossentropy(from_logits=False)

    # Define the optimizer

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.005,
    decay_steps=10000,
    decay_rate=0.96
    )

# Create an optimizer with the learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    # Compile the model
    classifier.compile(optimizer=optimizer, loss=criterion, metrics=["accuracy"])

    return classifier
