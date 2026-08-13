import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())  # 会输出 False
    print(torch.version.cuda)  # 会输出 None