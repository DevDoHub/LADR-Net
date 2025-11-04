#!/bin/bash
## 使用时先执行 oss login 登录

# 定义颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

mkdir -p /hy-tmp/synth-pedes
cd /hy-tmp/synth-pedes

# 定义所有文件名
files=(
    "Part1.tar.gz"
    "Part2.tar.gz"
    "Part3.tar.gz"
    "Part4.tar.gz"
    "Part5.tar.gz"
    "Part6.tar.gz"
    "Part7.tar.gz"
    "Part8.tar.gz"
    "Part9.tar.gz"
    "Part10.tar.gz"
    "Part11.tar.gz"
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
        tar -zxf "$file"
        rm -rf "$file"
        echo -e "         ${GREEN}✅ 完成: $file${NC}"
    else
        echo -e "         ${RED}❌ 下载失败: $file${NC}"
    fi
    echo
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}🎉 所有文件处理完成！${NC}"


