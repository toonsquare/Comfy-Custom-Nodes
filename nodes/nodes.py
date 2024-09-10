import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImagePalette
import deepl
import torchvision.transforms as transforms
from openai import OpenAI
import colorsys
from collections import Counter
import time
from server import PromptServer


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class PrintHelloWorld:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": False, "default": "Hello World"}),
        }
        }

    RETURN_TYPES = ()
    FUNCTION = "print_text"
    OUTPUT_NODE = True
    CATEGORY = "Toonsquare"

    def print_text(self, text):
        print(f"Tutorial Text : {text}")

        return {}


class DeepLTranslate:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "API_key": ("STRING", {"multiline": False, "default": "enter api key"}),
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True})
        }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "translate_to_english"
    OUTPUT_NODE = True
    CATEGORY = "Toonsquare"

    def translate_to_english(self, API_key, text):
        translator = deepl.Translator(API_key)

        result = translator.translate_text(text, target_lang="en-us")

        return (result,)


class ImageToGrayScale:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_to_grayscale"
    OUTPUT_NODE = True
    CATEGORY = "Toonsquare"

    def image_to_grayscale(self, image):
        to_tensor = transforms.ToTensor()

        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        image_gray = img.convert('L').convert('RGB')

        result = np.array(image_gray).astype(np.float32) / 255.0
        result = torch.from_numpy(result)[None,]
        return (result,)


class WallpaperPromptGenerator:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "deepL_key": ("STRING", {"multiline": False, "default": "enter api key"}),
            "openAI_key": ("STRING", {"multiline": False, "default": "enter api key"}),
            "text": ("STRING", {"multiline": True, "default": "light, shadow, neon, lines"}),
            "deepl_translation": (["on", "off"],),
        }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    OUTPUT_NODE = True
    CATEGORY = "Toonsquare"

    def generate_prompt(self, deepL_key, openAI_key, text, deepl_translation):
        if deepl_translation == "on":
            translator = deepl.Translator(deepL_key)
            input_text = translator.translate_text(text, target_lang="en-us").text
        else:
            input_text = text

        client = OpenAI(api_key=openAI_key)
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 이미지 생성 모델에 사용할 프롬프트를 제공하는 로봇입니다. 사용자는 당신에게 몇 가지 키워드를 제시할 것입니다. 당신은 그 키워드들을 사용해 월페이퍼 디자인의 이미지를 생성할 수 있는 프롬프트를 제공해야합니다. 아래 규칙을 따르며 해당 역할을 수행하세요. 1. 프롬프트는 영어로 작성해야 합니다. 필요시 영어로 번역하십시오. 2. 전후로 설명을 나열하거나 언급하지 마십시오. 3. 생성된 프롬프트는 'a blue light in the dark, a hologram inspired by Samuel F. B. Morse, tumblr, light and space, glowing tiny blue lines, quantum tracerwave'과 같은 문장 형식을 가져야합니다. 다시 말해, 전체적인 이미지를 설명하는 간단한 문장과 이미지에 대한 묘사나 풍부한 형용사 등이 쉼표로 구분된 형식입니다. 4. 생성된 프롬프트는 서로 상반된 내용의 단어들이 포함되면 안됩니다. 예를 들어, '구체적인'과 '간단한'은 동시에 있을 수 없습니다. 5.프롬프트를 통해 생성되는 이미지는 디지털 아트, 일러스트, 3D 렌러딩 등의 느낌이 나는 모바일 월페이퍼 이미지여야합니다. 이러한 이미지가 잘 생성될 수 있는 단어를 선택하는게 중요하다는 걸 명심하십시오. 6. 생성된 프롬프트는 매우 상세해야 하며, 약 100단어 길이여야 합니다.",
                },
                {
                    "role": "user",
                    "content": input_text,
                },
            ],
        )
        prompt = completion.choices[0].message.content

        return {"ui": {"prompt": (prompt,)}, "result": (prompt,)}
        # return (result,)


class PromptModifier:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "openAI_key": ("STRING", {"multiline": False, "default": "enter api key"}),
            "prompt": ("STRING", {"multiline": True, "default": "light, shadow, neon, lines"}),
            "modify": ("STRING", {"multiline": True, "default": "light, shadow, neon, lines"}),
        }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    OUTPUT_NODE = True
    CATEGORY = "Toonsquare"

    def generate_prompt(self, openAI_key, prompt, modify):
        client = OpenAI(api_key=openAI_key)
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 이미지 생성 모델에 사용할 프롬프트를 생성하는 로봇입니다. 사용자는 당신에게 프롬프트와 수정 지시 사항을 제시할 것입니다. 당신은 그것들을 사용해 수정된 새로운 프롬프트를 제공해야 합니다. 아래 규칙을 따르며 해당 역할을 수행하세요. 1. 프롬프트는 영어로 작성해야 합니다. 필요시 영어로 번역하십시오. 2. 이미지를 생성하기 전후로 설명을 나열하거나 언급하지 마십시오. 3. 생성된 프롬프트는 'a blue light in the dark, a hologram inspired by Samuel F. B. Morse, tumblr, light and space, glowing tiny blue lines, quantum tracerwave'과 같은 문장 형식을 가져야합니다. 다시 말해, 전체적인 이미지를 설명하는 간단한 문장과 이미지에 대한 묘사나 풍부한 형용사 등이 쉼표로 구분된 형식입니다. 4. 생성된 프롬프트는 사용자가 제공한 프롬프트에서 중요한 정보만을 가지도록 축약되어야하며, 수정 지시 사항 반영이 최우선시 되어야 합니다.",
                },
                {
                    "role": "user",
                    "content": "프롬프트 : " + prompt + ', 수정 지시 사항 : ' + modify,
                },
            ],
        )
        result = completion.choices[0].message.content

        return (result,)


class colorReferenceImage:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"width": ("INT", {"default": 1024, "min": 16, "step": 8}),
                             "height": ("INT", {"default": 1024, "min": 16, "step": 8}),
                             "hex": ("STRING", {"multiline": False, "default": "#000000"})}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_color_image"
    CATEGORY = "Toonsquare"

    def generate_color_image(self, width, height, hex):
        hex_color = hex.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        image = Image.new("RGB", (width, height), rgb_color)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)


class ImageToColorPalette:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_to_colorpalette"
    OUTPUT_NODE = True
    CATEGORY = "Toonsquare"

    def image_to_colorpalette(self, image):
        to_tensor = transforms.ToTensor()

        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img_rgb = img.convert('RGB')

        # 이미지에서 모든 픽셀 추출
        pixels = list(img_rgb.getdata())

        # 각 픽셀의 색상 빈도 계산
        counter = Counter(pixels)
        print(counter)

        # 가장 흔한 색상 10개를 선택
        top_colors = counter.most_common(10)

        # 색상과 빈도를 출력
        for color, count in top_colors:
            print(f"Color: {color}, Count: {count}")

        # 팔레트 이미지의 크기와 각 색상 블록의 크기 설정
        palette_width = 1024
        palette_height = 1024
        # 각 색상의 비율 계산
        total_pixels = sum(count for color, count in top_colors)
        color_ratios = [(color, count / total_pixels) for color, count in top_colors]

        # 새로운 팔레트 이미지를 생성
        palette_image = Image.new("RGB", (palette_width, palette_height))

        # 팔레트 이미지를 그리기
        current_x = 0
        for color, ratio in color_ratios:
            block_width = int(palette_width * ratio)
            for x in range(current_x, current_x + block_width):
                for y in range(palette_height):
                    palette_image.putpixel((x, y), color)
            current_x += block_width

        palette_image = np.array(palette_image).astype(np.float32) / 255.0
        result = torch.from_numpy(palette_image)[None,]
        # result = np.array(image_gray).astype(np.float32) / 255.0
        # result = torch.from_numpy(result)[None,]
        # result = to_tensor(image_gray)
        return (result,)


class GetResultAnything:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {},
                "optional": {"anything": (any, {}), }}

    RETURN_TYPES = ()
    FUNCTION = "get_result_anything"
    OUTPUT_NODE = True
    CATEGORY = "Toonsquare"

    def get_result_anything(self, anything):
        server = PromptServer.instance
        server.send_sync("status", {"status": "time to get result", "result": anything}, server.client_id)
        return ()

    def IS_CHANGED(s):
        return time.time()
