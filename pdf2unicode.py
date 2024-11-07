import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    # 打开 PDF 文件
    doc = fitz.open(file_path)
    text = ""

    # 遍历每一页并提取文本
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")  # "text" 输出 Unicode 文本

    doc.close()
    return text

def extract_text_as_unicode_escaped(file_path):
    doc = fitz.open(file_path)
    unicode_text = ""

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")

        # 将每个字符转换为 \u{xxxx} 格式
        escaped_text = "".join([f"\\u{{{ord(char):04X}}}" for char in page_text])
        unicode_text += escaped_text

    doc.close()
    return unicode_text

# 示例使用
pdf_path = "/Users/hzy/Downloads/test-7.pdf"
unicode_text = extract_text_from_pdf(pdf_path)
print(unicode_text)
