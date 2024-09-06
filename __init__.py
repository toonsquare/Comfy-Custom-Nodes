from .nodes.nodes import *

NODE_CLASS_MAPPINGS = { 
    "Print Hello World": PrintHelloWorld,
    "DeepL Translate": DeepLTranslate,
    "Image To GrayScale":ImageToGrayScale,
    "Wallpaper Prompt Generator": WallpaperPromptGenerator,
    "Prompt Modifier": PromptModifier,
    "Color Reference": colorReferenceImage,
    "Image To ColorPalette": ImageToColorPalette,
    "Get Result Anything":GetResultAnything
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]
