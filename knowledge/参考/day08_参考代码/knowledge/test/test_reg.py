import re

if __name__ == '__main__':
    line = "|      | ![image-20260601190834107](imgs/image-20260601190834107.png) | "
    img_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape("image-20260601190834107.png") + r".*?\)")
    t = img_pattern.search( line)
    print(t)