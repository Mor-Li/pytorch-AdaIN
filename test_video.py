import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import imageio
from torchvision import transforms

import net
from function import adaptive_instance_normalization, coral

import warnings
warnings.filterwarnings("ignore")

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_video', type=str,
                    help='File path to the content video')
parser.add_argument('--style_path', type=str,
                    help='File path to the style video or single image')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.mp4',
                    help='The extension name of the output video')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debug: Print device info
print(f"Using device: {device}")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Check input videos and models
assert (args.content_video), "Content video path is required!"
assert (args.style_path), "Style path is required!"
assert Path(args.vgg).exists(), f"VGG model not found at {args.vgg}"
assert Path(args.decoder).exists(), f"Decoder model not found at {args.decoder}"

content_path = Path(args.content_video)
style_path = Path(args.style_path)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

# Debug: Model loading
print("Loading VGG model...")
decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

# Open content video
content_video = cv2.VideoCapture(args.content_video)
if not content_video.isOpened():
    raise FileNotFoundError(f"Content video not found at {args.content_video}")

fps = int(content_video.get(cv2.CAP_PROP_FPS))
content_video_length = int(content_video.get(cv2.CAP_PROP_FRAME_COUNT))
output_width = int(content_video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(content_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Debug: Video properties
print(f"Content video FPS: {fps}, Frames: {content_video_length}, Width: {output_width}, Height: {output_height}")

assert fps != 0, 'FPS is zero, check content video path.'

pbar = tqdm(total=content_video_length)

if style_path.suffix in [".mp4", ".mpg", ".avi"]:
    style_video = cv2.VideoCapture(args.style_path)
    if not style_video.isOpened():
        raise FileNotFoundError(f"Style video not found at {args.style_path}")

    style_video_length = int(style_video.get(cv2.CAP_PROP_FRAME_COUNT))
    assert style_video_length == content_video_length, 'Frame mismatch between content and style video.'

    output_video_path = output_dir / f"{content_path.stem}_stylized_{style_path.stem}{args.save_ext}"
    writer = imageio.get_writer(output_video_path, mode='I', fps=fps)

    while True:
        ret, content_img = content_video.read()
        ret_style, style_img = style_video.read()

        if not ret or not ret_style:
            print("End of video reached.")
            break

        # Debug: Frame info
        print(f"Processing frame {pbar.n + 1}/{content_video_length}")

        try:
            content = content_tf(Image.fromarray(content_img))
            style = style_tf(Image.fromarray(style_img))

            if args.preserve_color:
                style = coral(style, content)

            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)

            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style, args.alpha)

            output = output.cpu().squeeze(0).numpy() * 255
            output = np.transpose(output, (1, 2, 0)).astype(np.uint8)
            output = cv2.resize(output, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
            writer.append_data(output)

        except Exception as e:
            print(f"Error processing frame {pbar.n + 1}: {e}")
            break

        pbar.update(1)

    style_video.release()
    content_video.release()

else:
    style_img = Image.open(style_path)
    output_video_path = output_dir / f"{content_path.stem}_stylized_{style_path.stem}{args.save_ext}"
    writer = imageio.get_writer(output_video_path, mode='I', fps=fps)

    while True:
        ret, content_img = content_video.read()
        if not ret:
            print("End of video reached.")
            break

        # Debug: Frame info
        print(f"Processing frame {pbar.n + 1}/{content_video_length}")

        try:
            content = content_tf(Image.fromarray(content_img))
            style = style_tf(style_img)

            if args.preserve_color:
                style = coral(style, content)

            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)

            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style, args.alpha)

            output = output.cpu().squeeze(0).numpy() * 255
            output = np.transpose(output, (1, 2, 0)).astype(np.uint8)
            output = cv2.resize(output, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
            writer.append_data(output)

        except Exception as e:
            print(f"Error processing frame {pbar.n + 1}: {e}")
            break

        pbar.update(1)

    content_video.release()

# Debug: Final message
print(f"Video stylization complete. Saved to {output_video_path}")