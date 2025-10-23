
# import torch

# # 第一步：加载完整的模型权重文件 (pass_vit_base_full.pth)
# checkpoint_path = 'pass_vit_base_full.pth'
# full_checkpoint = torch.load(checkpoint_path)

# # 第二步：创建一个映射字典，将旧的参数名称映射为新的参数名称
# name_mapping = {
#         'blocks.9.norm1.weight':'TransReid.0.norm1.weight',
# 'blocks.9.norm1.bias':'TransReid.0.norm1.bias',
# 'blocks.9.attn.qkv.weight':'TransReid.0.attn.qkv.weight',
# 'blocks.9.attn.qkv.bias':'TransReid.0.attn.qkv.bias',
# 'blocks.9.attn.proj.weight':'TransReid.0.attn.proj.weight',
# 'blocks.9.attn.proj.bias':'TransReid.0.attn.proj.bias',
# 'blocks.9.norm2.weight':'TransReid.0.norm2.weight',
# 'blocks.9.norm2.bias':'TransReid.0.norm2.bias',
# 'blocks.9.mlp.fc1.weight':'TransReid.0.mlp.fc1.weight',
# 'blocks.9.mlp.fc1.bias':'TransReid.0.mlp.fc1.bias',
# 'blocks.9.mlp.fc2.weight':'TransReid.0.mlp.fc2.weight',
# 'blocks.9.mlp.fc2.bias':'TransReid.0.mlp.fc2.bias',
# 'blocks.10.norm1.weight':'TransReid.1.norm1.weight',
# 'blocks.10.norm1.bias':'TransReid.1.norm1.bias',
# 'blocks.10.attn.qkv.weight':'TransReid.1.attn.qkv.weight',
# 'blocks.10.attn.qkv.bias':'TransReid.1.attn.qkv.bias',
# 'blocks.10.attn.proj.weight':'TransReid.1.attn.proj.weight',
# 'blocks.10.attn.proj.bias':'TransReid.1.attn.proj.bias',
# 'blocks.10.norm2.weight':'TransReid.1.norm2.weight',
# 'blocks.10.norm2.bias':'TransReid.1.norm2.bias',
# 'blocks.10.mlp.fc1.weight':'TransReid.1.mlp.fc1.weight',
# 'blocks.10.mlp.fc1.bias':'TransReid.1.mlp.fc1.bias',
# 'blocks.10.mlp.fc2.weight':'TransReid.1.mlp.fc2.weight',
# 'blocks.10.mlp.fc2.bias':'TransReid.1.mlp.fc2.bias',
# 'blocks.11.norm1.weight':'TransReid.2.norm1.weight',
# 'blocks.11.norm1.bias':'TransReid.2.norm1.bias',
# 'blocks.11.attn.qkv.weight':'TransReid.2.attn.qkv.weight',
# 'blocks.11.attn.qkv.bias':"TransReid.2.attn.qkv.bias",
# 'blocks.11.attn.proj.weight':'TransReid.2.attn.proj.weight',
# 'blocks.11.attn.proj.bias':'TransReid.2.attn.proj.bias',
# 'blocks.11.norm2.weight':'TransReid.2.norm2.weight',
# 'blocks.11.norm2.bias':'TransReid.2.norm2.bias',
# 'blocks.11.mlp.fc1.weight':'TransReid.2.mlp.fc1.weight',
# 'blocks.11.mlp.fc1.bias':'TransReid.2.mlp.fc1.bias',
# 'blocks.11.mlp.fc2.weight':'TransReid.2.mlp.fc2.weight',
# 'blocks.11.mlp.fc2.bias':'TransReid.2.mlp.fc2.bias' 
#     # 你可以继续添加其他需要映射的参数
# }

# # 第三步：创建一个新的状态字典，并使用映射字典修改参数名称
# new_state_dict = {}
# for old_name, new_name in name_mapping.items():
#     if old_name in full_checkpoint:
#         new_state_dict[new_name] = full_checkpoint[old_name]

# # 第四步：将新的状态字典保存到新的 .pth 文件 (checkpoint_tea.pth)
# new_checkpoint_path = 'checkpoint_tea.pth'
# torch.save(new_state_dict, new_checkpoint_path)

# print(f"新的检查点已保存到 {new_checkpoint_path}")
# import torch

# # 加载 checkpoint_tea.pth 文件
# checkpoint_path = 'checkpoint_tea.pth'
# checkpoint = torch.load(checkpoint_path)

# # 打开一个 txt 文件用于写入
# with open("checkpoint_parameters.txt", "w") as f:
#     # 遍历 checkpoint 中的所有参数名称和对应的值
#     for param_name, param_value in checkpoint.items():
#         # 写入参数名称
#         f.write(f"Parameter: {param_name}\n")
#         # 写入参数的数值（转换为 numpy 数组，以便更容易存储）
#         f.write(f"Values: {param_value.cpu().numpy()}\n\n")  # .cpu().numpy() 用于将张量转为 numpy 数组
# import torch

# # 第一步：加载完整的模型权重文件 (pass_vit_base_full.pth)
# checkpoint_path = 'pass_vit_base_full.pth'
# full_checkpoint = torch.load(checkpoint_path)

# # 第二步：加载现有的 checkpoint_tea.pth 文件（如果存在）
# new_checkpoint_path = 'checkpoint_tea.pth'
# try:
#     existing_checkpoint = torch.load(new_checkpoint_path)
# except FileNotFoundError:
#     existing_checkpoint = {}  # 如果文件不存在，则创建一个空字典

# # 第三步：创建一个映射字典，将旧的参数名称映射为新的参数名称
# name_mapping = {
# 'module.visual_encoder.blocks.9.norm1.weight':'TransReid.0.norm1.weight',
# 'module.visual_encoder.blocks.9.norm1.bias':'TransReid.0.norm1.bias',
# 'module.visual_encoder.blocks.9.attn.qkv.weight':'TransReid.0.attn.qkv.weight',
# 'module.visual_encoder.blocks.9.attn.qkv.bias':'TransReid.0.attn.qkv.bias',
# 'module.visual_encoder.blocks.9.attn.proj.weight':'TransReid.0.attn.proj.weight',
# 'module.visual_encoder.blocks.9.attn.proj.bias':'TransReid.0.attn.proj.bias',
# 'module.visual_encoder.blocks.9.norm2.weight':'TransReid.0.norm2.weight',
# 'module.visual_encoder.blocks.9.norm2.bias':'TransReid.0.norm2.bias',
# 'module.visual_encoder.blocks.9.mlp.fc1.weight':'TransReid.0.mlp.fc1.weight',
# 'module.visual_encoder.blocks.9.mlp.fc1.bias':'TransReid.0.mlp.fc1.bias',
# 'module.visual_encoder.blocks.9.mlp.fc2.weight':'TransReid.0.mlp.fc2.weight',
# 'module.visual_encoder.blocks.9.mlp.fc2.bias':'TransReid.0.mlp.fc2.bias',
# 'module.visual_encoder.blocks.10.norm1.weight':'TransReid.1.norm1.weight',
# 'module.visual_encoder.blocks.10.norm1.bias':'TransReid.1.norm1.bias',
# 'module.visual_encoder.blocks.10.attn.qkv.weight':'TransReid.1.attn.qkv.weight',
# 'module.visual_encoder.blocks.10.attn.qkv.bias':'TransReid.1.attn.qkv.bias',
# 'module.visual_encoder.blocks.10.attn.proj.weight':'TransReid.1.attn.proj.weight',
# 'module.visual_encoder.blocks.10.attn.proj.bias':'TransReid.1.attn.proj.bias',
# 'module.visual_encoder.blocks.10.norm2.weight':'TransReid.1.norm2.weight',
# 'module.visual_encoder.blocks.10.norm2.bias':'TransReid.1.norm2.bias',
# 'module.visual_encoder.blocks.10.mlp.fc1.weight':'TransReid.1.mlp.fc1.weight',
# 'module.visual_encoder.blocks.10.mlp.fc1.bias':'TransReid.1.mlp.fc1.bias',
# 'module.visual_encoder.blocks.10.mlp.fc2.weight':'TransReid.1.mlp.fc2.weight',
# 'module.visual_encoder.blocks.10.mlp.fc2.bias':'TransReid.1.mlp.fc2.bias',
# 'module.visual_encoder.blocks.11.norm1.weight':'TransReid.2.norm1.weight',
# 'module.visual_encoder.blocks.11.norm1.bias':'TransReid.2.norm1.bias',
# 'module.visual_encoder.blocks.11.attn.qkv.weight':'TransReid.2.attn.qkv.weight',
# 'module.visual_encoder.blocks.11.attn.qkv.bias':"TransReid.2.attn.qkv.bias",
# 'module.visual_encoder.blocks.11.attn.proj.weight':'TransReid.2.attn.proj.weight',
# 'module.visual_encoder.blocks.11.attn.proj.bias':'TransReid.2.attn.proj.bias',
# 'module.visual_encoder.blocks.11.norm2.weight':'TransReid.2.norm2.weight',
# 'module.visual_encoder.blocks.11.norm2.bias':'TransReid.2.norm2.bias',
# 'module.visual_encoder.blocks.11.mlp.fc1.weight':'TransReid.2.mlp.fc1.weight',
# 'module.visual_encoder.blocks.11.mlp.fc1.bias':'TransReid.2.mlp.fc1.bias',
# 'module.visual_encoder.blocks.11.mlp.fc2.weight':'TransReid.2.mlp.fc2.weight',
# 'module.visual_encoder.blocks.11.mlp.fc2.bias':'TransReid.2.mlp.fc2.bias' 
#     # 可以继续添加其他需要映射的参数
# }

# # 第四步：将新参数添加到现有的状态字典
# for old_name, new_name in name_mapping.items():
#     if old_name in full_checkpoint:
#         existing_checkpoint[new_name] = full_checkpoint[old_name]

# # 第五步：保存更新后的状态字典到 checkpoint_tea.pth
# torch.save(existing_checkpoint, new_checkpoint_path)

# print(f"新参数已添加，检查点保存到 {new_checkpoint_path}")
import torch

# 加载 checkpoint_tea.pth 文件
checkpoint_path = 'checkpoint_tea.pth'
try:
    checkpoint = torch.load(checkpoint_path)
    checkpoint = dict(checkpoint)
    # 打印所有参数的名称
    print("Checkpoint parameters:")
    for param_name in checkpoint.keys():
        # for i in param_name.keys():
        print(param_name)

except FileNotFoundError:
    print(f"文件 {checkpoint_path} 不存在。")
except Exception as e:
    print(f"加载文件时出现错误：{e}")
# import torch

# # 第一步：加载完整的模型权重文件 (pass_vit_base_full.pth)
# checkpoint_path = 'checkpoint_market.pth'
# full_checkpoint = torch.load(checkpoint_path)
# full_checkpoint = full_checkpoint['state_dict']
# # 第二步：加载现有的 checkpoint_tea.pth 文件（如果存在）
# new_checkpoint_path = 'checkpoint_tea.pth'
# try:
#     existing_checkpoint = torch.load(new_checkpoint_path)
# except FileNotFoundError:
#     existing_checkpoint = {}  # 如果文件不存在，则创建一个空字典

# # 第三步：创建一个映射字典，将旧的参数名称映射为新的参数名称
# name_mapping = {
# 'module.visual_encoder.blocks.9.norm1.weight':'TransReid.0.norm1.weight',
# 'module.visual_encoder.blocks.9.norm1.bias':'TransReid.0.norm1.bias',
# 'module.visual_encoder.blocks.9.attn.qkv.weight':'TransReid.0.attn.qkv.weight',
# 'module.visual_encoder.blocks.9.attn.qkv.bias':'TransReid.0.attn.qkv.bias',
# 'module.visual_encoder.blocks.9.attn.proj.weight':'TransReid.0.attn.proj.weight',
# 'module.visual_encoder.blocks.9.attn.proj.bias':'TransReid.0.attn.proj.bias',
# 'module.visual_encoder.blocks.9.norm2.weight':'TransReid.0.norm2.weight',
# 'module.visual_encoder.blocks.9.norm2.bias':'TransReid.0.norm2.bias',
# 'module.visual_encoder.blocks.9.mlp.fc1.weight':'TransReid.0.mlp.fc1.weight',
# 'module.visual_encoder.blocks.9.mlp.fc1.bias':'TransReid.0.mlp.fc1.bias',
# 'module.visual_encoder.blocks.9.mlp.fc2.weight':'TransReid.0.mlp.fc2.weight',
# 'module.visual_encoder.blocks.9.mlp.fc2.bias':'TransReid.0.mlp.fc2.bias',
# 'module.visual_encoder.blocks.10.norm1.weight':'TransReid.1.norm1.weight',
# 'module.visual_encoder.blocks.10.norm1.bias':'TransReid.1.norm1.bias',
# 'module.visual_encoder.blocks.10.attn.qkv.weight':'TransReid.1.attn.qkv.weight',
# 'module.visual_encoder.blocks.10.attn.qkv.bias':'TransReid.1.attn.qkv.bias',
# 'module.visual_encoder.blocks.10.attn.proj.weight':'TransReid.1.attn.proj.weight',
# 'module.visual_encoder.blocks.10.attn.proj.bias':'TransReid.1.attn.proj.bias',
# 'module.visual_encoder.blocks.10.norm2.weight':'TransReid.1.norm2.weight',
# 'module.visual_encoder.blocks.10.norm2.bias':'TransReid.1.norm2.bias',
# 'module.visual_encoder.blocks.10.mlp.fc1.weight':'TransReid.1.mlp.fc1.weight',
# 'module.visual_encoder.blocks.10.mlp.fc1.bias':'TransReid.1.mlp.fc1.bias',
# 'module.visual_encoder.blocks.10.mlp.fc2.weight':'TransReid.1.mlp.fc2.weight',
# 'module.visual_encoder.blocks.10.mlp.fc2.bias':'TransReid.1.mlp.fc2.bias',
# 'module.visual_encoder.blocks.11.norm1.weight':'TransReid.2.norm1.weight',
# 'module.visual_encoder.blocks.11.norm1.bias':'TransReid.2.norm1.bias',
# 'module.visual_encoder.blocks.11.attn.qkv.weight':'TransReid.2.attn.qkv.weight',
# 'module.visual_encoder.blocks.11.attn.qkv.bias':"TransReid.2.attn.qkv.bias",
# 'module.visual_encoder.blocks.11.attn.proj.weight':'TransReid.2.attn.proj.weight',
# 'module.visual_encoder.blocks.11.attn.proj.bias':'TransReid.2.attn.proj.bias',
# 'module.visual_encoder.blocks.11.norm2.weight':'TransReid.2.norm2.weight',
# 'module.visual_encoder.blocks.11.norm2.bias':'TransReid.2.norm2.bias',
# 'module.visual_encoder.blocks.11.mlp.fc1.weight':'TransReid.2.mlp.fc1.weight',
# 'module.visual_encoder.blocks.11.mlp.fc1.bias':'TransReid.2.mlp.fc1.bias',
# 'module.visual_encoder.blocks.11.mlp.fc2.weight':'TransReid.2.mlp.fc2.weight',
# 'module.visual_encoder.blocks.11.mlp.fc2.bias':'TransReid.2.mlp.fc2.bias' 
#     # 可以继续添加其他需要映射的参数
# }

# # 第四步：将新参数添加到现有的状态字典
# for old_name, new_name in name_mapping.items():
#     if old_name in full_checkpoint:
#         existing_checkpoint[new_name] = full_checkpoint[old_name]

# # 第五步：保存更新后的状态字典到 checkpoint_tea.pth
# torch.save(existing_checkpoint, new_checkpoint_path)

# print(f"新参数已添加，检查点保存到 {new_checkpoint_path}")
