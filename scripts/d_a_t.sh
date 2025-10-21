#!/bin/bash
## 使用时先执行 oss login 登录

# 定义颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

mkdir -p /hy-tmp/LUPerson
cd /hy-tmp/LUPerson


# 定义所有文件名
files=(
    "LUPerson.zip.001.tar"
    "LUPerson.zip.002.tar"
    "LUPerson.zip.003.tar"
    "LUPerson.zip.004.tar"
    "LUPerson.zip.005.tar"
    "LUPerson.zip.006.tar"
    "LUPerson.zip.007.tar"
    "LUPerson.zip.008.tar"
    "LUPerson.zip.009.tar"
    "LUPerson.zip.010.tar"
    "LUPerson.zip.011.tar"
    "LUPerson.zip.012.tar"
    "LUPerson.zip.013.tar"
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
        echo -e "         ✅ 下载完成 → 🗜️  tar解包中..."
        tar -xf "$file"
        rm -rf "$file"
        echo -e "         ${GREEN}✅ 完成: $file${NC}"
    else
        echo -e "         ${RED}❌ 下载失败: $file${NC}"
    fi
    echo
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}🎉 所有文件处理完成！${NC}"

