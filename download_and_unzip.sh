#!/bin/bash
## 使用时先执行 oss login 登录

# 定义颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

mkdir -p /hy-tmp/sda
cd /hy-tmp/sda

# 定义所有文件名
files=(
    "c_0_5000.zip"
    "c_5000_10000.zip"
    "c_10000_15000.zip"
    "c_15000_20000.zip"
    "c_20000_25000.zip"
    "c_25000_30000.zip"
    "c_30000_35000.zip"
    "c_35000_40000.zip"
    "c_40000_45000.zip"
    "c_45000_50000.zip"
    "c_50000_55000.zip"
    "c_55000_60000.zip"
    "c_60000_64000.zip"
    "c_64000_70000.zip"
    "sda.zip"
)

total=${#files[@]}
current=0

echo -e "${BLUE}📦 开始批量下载和解压 OSS 文件 (共 $total 个文件)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 循环处理每个文件
for file in "${files[@]}"; do
    current=$((current + 1))
    echo -e "${YELLOW}[$current/$total]${NC} 🔄 处理中: ${file}"
    
    oss cp "oss://$file" ./
    if [ $? -eq 0 ]; then
        echo -e "         ✅ 下载完成 → 🗜️  解压中..."
        unzip -q "$file"
        rm -rf "$file"
        echo -e "         ${GREEN}✅ 完成: $file${NC}"
    else
        echo -e "         ${RED}❌ 下载失败: $file${NC}"
    fi
    echo
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}🎉 所有文件处理完成！${NC}"









# 简单版本
# -----------------------------------------

# #!/bin/bash

# mkdir -p /hy-tmp/sda
# cd /hy-tmp/sda

# # 定义所有文件名
# files=(
#     "c_0_5000.zip"
#     "c_5000_10000.zip"
#     "c_10000_15000.zip"
#     "c_15000_20000.zip"
#     "c_20000_25000.zip"
#     "c_25000_30000.zip"
#     "c_30000_35000.zip"
#     "c_35000_40000.zip"
#     "c_40000_45000.zip"
#     "c_45000_50000.zip"
#     "c_50000_55000.zip"
#     "c_55000_60000.zip"
#     "c_60000_64000.zip"
#     "c_64000_70000.zip"
#     "sda.zip"
# )

# # 循环处理每个文件
# for file in "${files[@]}"; do
#     echo "[INFO] ==> 开始下载: $file"
#     oss cp "oss://$file" ./
#     if [ $? -eq 0 ]; then
#         echo "[INFO] ==> 完成下载: $file"
#         echo "[INFO] ==> 开始解压: $file"
#         unzip -q "$file"
#         echo "[INFO] ==> 完成解压: $file"
#         echo "[INFO] ==> 开始删除: $file"
#         rm -rf "$file"
#         echo "[INFO] ==> 完成删除: $file"
#     else
#         echo "[ERROR] ==> 下载失败: $file"
#     fi
# done

# echo "[INFO] ==> 所有文件处理完成"
