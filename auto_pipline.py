import subprocess
import shutil
import os
import time

def run_script(script_name):
    print(f"🚀 [啟動] 正在執行: {script_name} ...")
    # 使用 subprocess 呼叫 python 執行檔案
    result = subprocess.run(["python", script_name])
    
    if result.returncode != 0:
        print(f"❌ [錯誤] {script_name} 執行失敗！腳本已中斷。")
        exit(1) # 如果報錯就立刻停止，保護後續流程
    print(f"✅ [完成] {script_name} 執行完畢！\n")

def clean_folder(folder_path):
    print(f"🗑️ [清理] 準備清空資料夾: {folder_path}")
    if os.path.exists(folder_path):
        try:
            # 整個資料夾刪除，然後立刻建一個空的回來，保證 100% 乾淨
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
            print(f"✨ [成功] 已徹底清空 {folder_path}！")
        except Exception as e:
            print(f"⚠️ [警告] 清理 {folder_path} 時發生錯誤: {e}")
    else:
        # 如果資料夾本來就不存在，就順手建一個
        os.makedirs(folder_path)
        print(f"✨ [建立] 找不到原資料夾，已自動建立全新的 {folder_path}。")

if __name__ == "__main__":
    print("========================================")
    print("      🤖 AI 裁判全自動訓練流水線啟動      ")
    print("========================================\n")

    start_time = time.time()
    
    # 目標資料夾：存放要給裁判看的假圖
    target_fake_dir = "./discriminator_dataset/fake"
    target_fake_dir2 = "./augmented_fake"

    # --- 步驟 0：拔除雜草 (清理舊檔案) ---
    print("🧹 【步驟 0】確保環境乾淨...")
    clean_folder(target_fake_dir)
    clean_folder(target_fake_dir2)
    print("")

    # --- 步驟 1：抽樣假資料 ---
    print("🎲 【步驟 1】開始抽取新的假圖...")
    run_script("sample_fake_data.py")

    # --- 步驟 2：訓練 AI 裁判 ---
    print("⚖️ 【步驟 2】呼叫 AI 裁判開庭...")
    run_script("train_discriminator.py")

#    # --- 步驟 3：毀屍滅跡 (刪除原檔案) ---
#    print("🔥 【步驟 3】訓練結束，執行毀屍滅跡...")
#    clean_folder(target_fake_dir)

    end_time = time.time()
    print("\n========================================")
    print(f"🎉 全部任務流水線完美執行完畢！總耗時: {end_time - start_time:.2f} 秒")
    print("========================================")