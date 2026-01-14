import fitz  # PyMuPDF
from natsort import natsorted
import os
import base64
import json

def extract_pdf_paths_to_svg(pdf_path, output_dir):
    # 打开PDF文件
    document = fitz.open(pdf_path)

    # 遍历PDF的每一页
    for page_number in range(len(document)):
        page = document.load_page(page_number)

        # 提取页面的SVG内容
        svg_content = page.get_svg_image(text_as_path=True)

        # 保存SVG文件
        svg_filename = f"{output_dir}/page_{page_number + 1}.svg"
        with open(svg_filename, "w") as svg_file:
            svg_file.write(svg_content)

        print(f"Saved: {svg_filename}")

    print("All pages have been extracted to SVG files.")

# # 使用示例
# # 去除276， 278
# extract_pdf_paths_to_svg(pdf_file, output_directory)

def rename_files_in_folder(folder_path):
    # 获取文件夹中的所有文件名，并排除 .DS_Store 文件
    files = [f for f in os.listdir(folder_path) if f != '.DS_Store']

    # 使用 natsorted 进行自然排序
    sorted_files = natsorted(files)

    # 遍历排序后的文件，并按序号重命名
    for index, filename in enumerate(sorted_files):
        # 获取文件的完整路径
        old_path = os.path.join(folder_path, filename)

        # 构建新的文件名，格式如 "1.ext", "2.ext", ...
        file_extension = os.path.splitext(filename)[1]
        new_filename = f"{index + 1}{file_extension}"
        new_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_filename}'")

# rename_files_in_folder('/Users/hzy/Downloads/symbolizer/output_svgs')


def encode_svg_files_to_base64(folder_path, output_file):
    svg_data = {}

    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是.svg文件
        if filename.endswith('.svg'):
            file_path = os.path.join(folder_path, filename)

            # 打开并读取文件内容
            with open(file_path, 'rb') as file:
                encoded_content = base64.b64encode(file.read()).decode('utf-8')

            name = filename.removesuffix(".svg")
            key = name.replace("page_", "")
            svg_data[key] = encoded_content

    # 将文件名和编码内容写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(svg_data, json_file, ensure_ascii=False, indent=4)

    print(f"所有文件已成功编码并保存至 {output_file}")

if __name__ == "__main__":
    pdf_file = "/Users/hzy/Downloads/all_latex_symbols.pdf"
    folder_path = './output_svgs'  # 指定文件夹路径
    output_file = './latex_symbol_svgs.json'       # 指定输出的JSON文件名称

    extract_pdf_paths_to_svg(pdf_file, folder_path)
    encode_svg_files_to_base64(folder_path, output_file)
