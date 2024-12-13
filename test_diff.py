import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import net
from function import adaptive_instance_normalization, coral
import matplotlib.pyplot as plt  # 新增导入 matplotlib 库
import numpy as np  # 用于处理图像数据

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
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
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
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
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
do_interpolation = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)
# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]
# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]
decoder = net.decoder
vgg = net.vgg
decoder.eval()
vgg.eval()
decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)
decoder.to(device)
content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)
for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))
    else:  # process one content and one style
        for style_path in style_paths:
            # 加载并转换内容和风格图像
            content_image = Image.open(str(content_path))
            style_image = Image.open(str(style_path))
            content = content_tf(content_image)
            style = style_tf(style_image)
            if args.preserve_color:
                style = coral(style, content)
            style_tensor = style.unsqueeze(0).to(device)
            content_tensor = content.unsqueeze(0).to(device)
            with torch.no_grad():
                output_tensor = style_transfer(vgg, decoder, content_tensor, style_tensor,
                                               args.alpha)
            output = output_tensor.cpu().squeeze(0)
            # 将 Tensor 转换为 numpy 数组，用于绘图
            content_np = content.numpy().transpose(1, 2, 0)
            style_np = style.numpy().transpose(1, 2, 0)
            output_np = output.numpy().transpose(1, 2, 0)
            # 由于图像数据范围在 [0,1]，确保数据在此范围内
            content_np = np.clip(content_np, 0, 1)
            style_np = np.clip(style_np, 0, 1)
            output_np = np.clip(output_np, 0, 1)
            # 创建一行三列的子图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # 显示内容图像
            axes[0].imshow(content_np)
            axes[0].set_title('Content Image')
            axes[0].axis('off')
            # 显示风格图像
            axes[1].imshow(style_np)
            axes[1].set_title('Style Image')
            axes[1].axis('off')
            # 显示输出图像
            axes[2].imshow(output_np)
            axes[2].set_title('Output Image')
            axes[2].axis('off')
            # 调整布局并保存图像
            plt.tight_layout()
            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            plt.savefig(str(output_name))
            plt.close()
