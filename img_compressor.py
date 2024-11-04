from PIL import Image

# 加载图片
img = Image.open("./imgs/alpha.png")
# img.show()
# 转换为灰度图像
# 检查是否有 alpha 通道
if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
    # 创建 RGBA 模式的白色背景
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    # 将图片合并到白色背景上
    img = Image.alpha_composite(background, img.convert("RGBA"))

img_gray = img.convert("L")

# 二值化处理
threshold = 128
img_binary = img_gray.point(lambda x: 255 if x > threshold else 0, mode='1')

# 调整大小为32x32，使用 LANCZOS 替代 ANTIALIAS
img_resized = img_binary.resize((32, 32), Image.LANCZOS)

# 保存或显示结果
img_resized.save("output.png")
img_resized.show()
