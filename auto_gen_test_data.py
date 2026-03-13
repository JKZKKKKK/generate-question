import os
import sys
import subprocess
import shutil

# ================= 💡 超強防呆：自動導航系統 =================
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"📂 已自動切換工作目錄至: {os.getcwd()}")

py_exe = sys.executable

# ================= 路徑設定區 =================
gen_script = "generate_test_data.py"
flatten_script = "flatten_and_rename.py"

raw_images_dir = os.path.join("data", "landmarks") 
generated_test_dir = os.path.join("data", "test")    # 初步生成的分類圖片
final_test_dir = os.path.join("data", "fin_test")    # 最終考卷

zip_fin_test_name = os.path.join("data", "fin_test") 
zip_test_name = os.path.join("data", "test")

# ================= 🧹 新增：自動大掃除 (清空舊資料) =================
print("\n🧹 正在清理上次生成的舊資料...")

# 1. 清理舊資料夾
for d in [generated_test_dir, final_test_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)  # 整個資料夾連同裡面的檔案一起砍掉
        print(f"🗑️  已刪除舊資料夾: {d}")

# 2. 清理舊壓縮檔
for f in [f"{zip_fin_test_name}.zip", f"{zip_test_name}.zip"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"🗑️  已刪除舊壓縮檔: {f}")

print("✨ 清理完畢！準備開始生成全新測資。\n")


# ================= 指令清單 =================
# ================= 指令清單 =================
commands = [
    [
        py_exe, gen_script,
        "--input_dir", raw_images_dir,
        "--output_dir", generated_test_dir,
        "--config", "config.json"  # 👈 告訴程式去讀取這份設定檔
    ],
    [
        py_exe, flatten_script,
        "--input_dir", generated_test_dir,
        "--output_dir", final_test_dir,
        "--seed", "42"
    ]
]

print("🚀 開始自動化生成測資 (出考卷) 流程...")
print(f"🐍 使用的 Python 核心為: {py_exe}")

# 依序執行指令
for cmd in commands:
    cmd_str = " ".join(cmd)
    print(f"\n[{'='*50}]")
    print(f"👉 正在執行:\n{cmd_str}")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ 錯誤發生！中斷執行。請檢查上方的錯誤訊息。")
        break
else:
    print(f"\n[{'='*50}]")
    print(f"🎉 測資一鍵生成完畢！")
    
    # ================= 自動壓縮打包步驟 =================
    print(f"\n📦 正在將 [fin_test] 連同資料夾外殼壓縮成 ZIP 檔，請稍候...")
    shutil.make_archive(zip_fin_test_name, 'zip', root_dir='data', base_dir='fin_test')
    print(f"✅ 完成！檔案已儲存為: {zip_fin_test_name}.zip")
    
    print(f"\n📦 正在將 [test] 連同資料夾外殼壓縮成 ZIP 檔，請稍候...")
    shutil.make_archive(zip_test_name, 'zip', root_dir='data', base_dir='test')
    print(f"✅ 完成！檔案已儲存為: {zip_test_name}.zip")
    
    print(f"\n🎊 所有任務圓滿結束！解壓縮後會有整齊的資料夾囉！")