import os
import itertools
import argparse

def main():
    parser = argparse.ArgumentParser(description='Batch Style Transfer')
    parser.add_argument('--content_dir', type=str, default='input/content', help='Directory of content images')
    parser.add_argument('--style_dir', type=str, default='input/style', help='Directory of style images')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output images')
    parser.add_argument('--num_pairs', type=int, default=20, help='Number of content-style pairs to process')
    args = parser.parse_args()

    # 获取内容图像和风格图像列表
    content_images = [os.path.join(args.content_dir, f) for f in os.listdir(args.content_dir)
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
    style_images = [os.path.join(args.style_dir, f) for f in os.listdir(args.style_dir)
                    if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 生成所有可能的内容-风格图像组合
    combinations = list(itertools.product(content_images, style_images))

    # 如果指定的组合数量超过总组合数量，则调整为总组合数量
    if args.num_pairs > len(combinations):
        args.num_pairs = len(combinations)

    # 选择前 N 个组合
    combinations = combinations[:args.num_pairs]

    for idx, (content_img, style_img) in enumerate(combinations):
        command = f"python test_diff.py --content {content_img} --style {style_img} --output {args.output_dir}"
        print(f"Processing {idx+1}/{args.num_pairs}: Content - {content_img}, Style - {style_img}")
        os.system(command)

if __name__ == '__main__':
    main()
