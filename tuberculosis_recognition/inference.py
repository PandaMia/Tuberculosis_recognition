import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms

from tuberculosis_recognition.paths import DATA_PARAMS_PATH, EXAMPLES_DIR, MODEL_WEIGHTS_PATH


class ImageRecognition(object):

    def __init__(self, model):
        self.params_file = DATA_PARAMS_PATH
        img_shape, mean, std = self.load_params()
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(img_shape),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )
        self.width = 500
        self.model = model
        self.model.eval()
        self.draw_config = {
            "font_size": 50,
            "rect_top": 30,
            "rect_side_padding": 25,
            "rect_inner_padding": 12,
            "line_gap": 8,
        }

    def __call__(self, bytes_data):
        input = self.prepare_image(bytes_data)
        pred, prob = self.recognize(input)
        result_image = self.postprocess(bytes_data, pred, prob)
        return result_image

    def load_params(self):
        with open(self.params_file, "r") as f:
            params = json.loads(f.read())
        img_shape = params["img_shape"]
        mean = torch.tensor(params["mean"])
        std = torch.tensor(params["std"])
        return img_shape, mean, std

    def prepare_image(self, bytes_data):
        image = Image.open(io.BytesIO(bytes_data))
        image = self.image_transforms(image).unsqueeze(0)
        return image

    def load_font(self, size):
        font_candidates = (
            "Arial.ttf",
            "arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )

        for font_path in font_candidates:
            try:
                return ImageFont.truetype(str(Path(font_path)), size)
            except OSError:
                continue

        return ImageFont.load_default(size=size)

    def fit_font(self, draw, text, max_width, max_size, min_size=16):
        for size in range(max_size, min_size - 1, -2):
            font = self.load_font(size)
            bbox = draw.textbbox((0, 0), text, font=font)
            if bbox[2] - bbox[0] <= max_width:
                return font, bbox

        font = self.load_font(min_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        return font, bbox

    def recognize(self, input):
        with torch.no_grad():
            output = self.model(input)
            prob = np.round(torch.sigmoid(output.detach()).squeeze(1).cpu().item(), 4)
            pred = np.round(prob)
            percent_prob = np.round(100 * prob, 2)
        return pred, percent_prob

    def postprocess(self, bytes_data, pred, prob):
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        image = self.scale_image(image)
        if pred:
            title = "Tuberculosis!"
            confidence_text = f"Confidence: {prob}%"
            banner_color = (208, 118, 119, 150)
            image[..., 0] += 80
        else:
            title = "Clear!"
            confidence_text = f"Confidence: {100-prob}%"
            banner_color = (120, 199, 119, 150)
            image[..., 1] += 80

        image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        im_draw = ImageDraw.Draw(image, "RGBA")

        side_padding = self.draw_config["rect_side_padding"]
        inner_padding = self.draw_config["rect_inner_padding"]
        max_text_width = image.width - 2 * (side_padding + inner_padding)

        title_font, title_bbox = self.fit_font(
            im_draw,
            title,
            max_text_width,
            self.draw_config["font_size"],
        )
        confidence_font, confidence_bbox = self.fit_font(
            im_draw,
            confidence_text,
            max_text_width,
            self.draw_config["font_size"],
        )

        title_height = title_bbox[3] - title_bbox[1]
        confidence_height = confidence_bbox[3] - confidence_bbox[1]
        rect_top = self.draw_config["rect_top"]
        rect_bottom = (
            rect_top
            + inner_padding * 2
            + title_height
            + self.draw_config["line_gap"]
            + confidence_height
        )
        rect_shape = [(0, rect_top), (image.width, rect_bottom)]

        title_width = title_bbox[2] - title_bbox[0]
        title_x = (image.width - title_width) // 2
        title_y = rect_top + inner_padding - title_bbox[1]
        confidence_x = side_padding
        confidence_y = title_y + title_height + self.draw_config["line_gap"]

        im_draw.rectangle(rect_shape, fill=banner_color)
        im_draw.text((title_x, title_y), title, fill=(255, 255, 255), font=title_font)
        im_draw.text((confidence_x, confidence_y), confidence_text, fill=(255, 255, 255), font=confidence_font)
        return image

    def scale_image(self, image):
        w, h = image.size
        coef = self.width / w
        image = image.resize((int(w * coef), int(h * coef)))
        image = np.array(image, dtype=np.int32)
        return image


if __name__ == "__main__":
    from tuberculosis_recognition.cnn_model import CNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = MODEL_WEIGHTS_PATH
    model = CNN()
    model.load_state_dict(torch.load(weights, map_location=device))

    image_path = EXAMPLES_DIR / "Normal" / "1.png"
    with open(image_path, "rb") as file:
        bytes_data = file.read()

    recognizer = ImageRecognition(model)
    image = recognizer(bytes_data)
    image.show()
