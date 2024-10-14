import re


def generate_toc(md_content):
    toc = []
    headings = re.findall(r'^(#{1,6})\s+(.+)', md_content, re.MULTILINE)
    for heading in headings:
        level = len(heading[0])
        title = heading[1]
        # anchor = re.sub(r'[^\w\s-]', '', title).strip().lower()
        # anchor = re.sub(r'[-\s]+', '-', anchor)
        indent = ' ' * 2 * (level - 1)
        toc.append(f'{indent}<a href="#{title}">{title}</a><br/>\n')
    return '<nav>\n' + ''.join(toc) + '</nav>'


def process_markdown_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    toc = generate_toc(md_content)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(toc)


if __name__ == "__main__":
    input_path = 'input.md'  # 输入的Markdown文件路径
    output_path = 'output.html'  # 输出的HTML文件路径
    process_markdown_file(input_path, output_path)