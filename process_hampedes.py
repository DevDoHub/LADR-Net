import json

# 读原文件
with open("/root/SOLIDER-REID-PRO/data/HAM_pedes/random100w_2HAMcaptions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = {}
for idx, (filename, captions) in enumerate(data.items(), 1):
    new_data[filename] = {
        "captain": captions,
        "id": idx
    }

# 保存新文件
with open("/root/SOLIDER-REID-PRO/data/HAM_pedes/random100w_2HAMcaptions_new.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
