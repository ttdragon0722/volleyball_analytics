# game_state_trainer：微調 VideoMAE 比賽狀態模型

用 `videomae_labeler` 標好的影片，微調 SERVICE / PLAY / NO-PLAY 模型。
標註直接從 `db/annotations.db` 讀取，不需要先匯出成檔案。

## 使用方式

```powershell
cd D:\coding\volleyball_analytics
$env:PYTHONPATH = "src"

# 1. 看資料庫裡有哪些影片已經有標註
.venv\Scripts\python.exe -m game_state_trainer --list

# 2. 訓練（訓練和驗證一定要用不同的影片；兩邊都可以放多部）
.venv\Scripts\python.exe -m game_state_trainer `
    --train "D:/user/Downloads/2026_10_29 3-1_1080p60.mp4" `
    --val   "D:/user/Downloads/2026_10_29 2-2_1080p60.mp4"

# 3. 只評估某個模型（不訓練），例如比較兩個版本
.venv\Scripts\python.exe -m game_state_trainer --evaluate output\game_state_models\v1\best `
    --val "D:/user/Downloads/2026_10_29 2-2_1080p60.mp4"
```

訓練完成後會印出最好的模型放在哪裡，預設是 `output\game_state_models\<時間>\best`。這個資料夾可以直接拿來用：

```powershell
# 用新模型預標（在標註工具裡按「模型預標」）
cd src
..\.venv\Scripts\python.exe -m videomae_labeler --model ..\output\game_state_models\v1\best

# 用新模型剪輯比賽片段
.venv\Scripts\python.exe -m game_state_clipper "影片.mp4" --model output\game_state_models\v1\best
```

## 運作方式

1. **影格快取**：每部影片第一次使用時會整部解碼一次，存成 224×224 的 JPEG（`output\game_state_cache\`）。
   一部 20 分鐘的 1080p60 影片大約需要幾分鐘、200 MB 左右。之後再訓練就直接讀快取。
   影片檔案有變動時，會自動重建快取。
2. **取樣方式和推論時一樣**：每個樣本是 1 秒的窗口，平均取 16 幀，整張畫面直接壓成 224×224，再做 ImageNet 正規化。
3. **平衡取樣**：每個 epoch 中，service／play／no-play 各抽 `--samples-per-class` 個窗口（預設 400 個），
   起點隨機，並且會做左右翻轉、裁切、亮度和對比的隨機變化。所以 service 雖然少，也不會被模型忽略。
   只有多數類別至少佔窗口 75% 的窗口才會被拿來訓練（`--min-purity`）。
4. **驗證方式和推論時一樣**：把驗證影片切成連續、不重疊的 1 秒窗口，每個窗口用多數類別當答案。
   開始訓練前會先評估原本的模型當作**基準**，訓練完就能比較有沒有進步。
5. **從 `src/weights/game_state` 微調，但分類層會重新初始化**：
   原本模型的主幹網路沒有問題，但最後一層分類層的權重異常（最大到 7000 左右，輸出的信心度永遠接近 1.00），
   所以只保留主幹，分類層重新學。主幹的學習率是 3e-5，分類層是 1e-3。要沿用原本的分類層，可以加 `--keep-head`。
6. **只保留最好的 epoch**：以驗證集的 macro-F1（三個類別 F1 的平均）來選，
   存到 `best\`（HuggingFace 格式）。完整的訓練紀錄寫在 `report.json`。

## 常用參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `--epochs` | 8 | 訓練幾輪 |
| `--batch-size` / `--grad-accum` | 4 / 4 | 等效 batch 是 16 |
| `--lr` | 3e-5 | 主幹網路的學習率 |
| `--samples-per-class` | 400 | 每個 epoch 每個類別抽幾個窗口 |
| `--min-purity` | 0.75 | 訓練窗口裡多數類別至少要佔多少比例 |
| `--grad-checkpointing` | 關 | 出現 `CUDA out of memory` 時打開（比較慢，但省記憶體） |
| `--skip-baseline` | 關 | 不先評估原本的模型 |
| `--keep-head` | 關 | 沿用原模型的分類層 |

## 怎麼看結果

- **macro-F1**：三個類別 F1 的平均，是挑選模型的主要依據。
- **service recall**：真正的發球裡，有幾成被認出來。service 最難學，最值得注意。
- **混淆矩陣**：每一列是正確答案，每一行是模型的判斷。例如 `service` 那一列在 `play` 那一行的數字很大，
  代表發球常被誤判成 play，可能是 service 的起點或終點標得不一致。

> 驗證集只有一部影片時，分數會有一定的波動。標了更多影片之後，驗證集最好也放 2 部以上。
