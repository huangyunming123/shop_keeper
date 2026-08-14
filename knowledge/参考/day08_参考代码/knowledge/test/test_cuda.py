import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())  # 应该输出 True
    print(torch.version.cuda)  # 输出 12.8 或 11.8
    # print(torch.cuda.get_device_name(0))  # 显示你的显卡名字