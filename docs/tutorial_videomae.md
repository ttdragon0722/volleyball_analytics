# VideoMAE 比賽狀態模型微調教學

用自己拍的排球影片，微調 SERVICE / PLAY / NO-PLAY 比賽狀態分類模型（VideoMAE）。

> **為什麼要微調？** 專案內建的權重（`src/weights/game_state`）是用國際賽轉播畫面（端線後方、高機位）訓練的。
> 拿去跑側面低機位的業餘比賽影片時，90% 以上的時間都被判成 `no-play`，而且信心度是 1.00，等於完全判錯卻毫不猶豫。
> 只要用同一個場地、同一種機位的影片標註幾百秒的資料，就能大幅改善。

---

## 0. 整體流程

```
① 標註                    ② 訓練                     ③ 換上新模型
videomae_labeler     →    game_state_trainer    →    videomae_labeler / game_state_clipper
（按住 1/2/3，Ctrl+S）    （直接讀 annotations.db，   --model <新模型>/best
                          從排球權重微調）
```

| 步驟 | 狀態 |
|---|---|
| ① 標註 | ✅ 網頁版 `videomae_labeler`（`src/videomae_labeler/README.md`）。舊的 Qt 版 vb_annotator 也能用（本文第 1～2 節） |
| ② 訓練 | ✅ `game_state_trainer`（`src/game_state_trainer/README.md`），不需要先匯出成檔案 |
| ③ 推論 | ✅ 訓練輸出的就是 HuggingFace 格式的資料夾，可以直接用 `--model` 指定 |

標註規範見 [`videomae_readme.md`](videomae_readme.md)。

---

## 1. 啟動標註工具

在 PowerShell 執行（專案根目錄 `D:\coding\volleyball_analytics`）：

```powershell
cd D:\coding\volleyball_analytics\src\vb_gui\vb_annotator
$env:PYTHONPATH = "D:\coding\volleyball_analytics;D:\coding\volleyball_analytics\src"
..\..\..\.venv\Scripts\python.exe main.py
```

- 標註結果存在 `D:\coding\volleyball_analytics\db\annotations.db`。第一次啟動會自動建立。
- 要換資料庫位置，可以先設定環境變數 `$env:VB_ANNOTATIONS_DB = "D:\其他路徑\xxx.db"`。
- **儲存與接續**：
  - Mark End、Range… 預標、Delete Selected 都會**立刻**寫入資料庫。
  - 在 Timeline 上拖曳的修改，要按 Apply Changes 或 `Ctrl+S` 才會存。如果直接關閉視窗，程式會先詢問要不要儲存。
  - 程式會記住上次停在哪部影片的第幾幀（存在 `db\annotations_session.json`，關閉時和每 30 秒各存一次）。下次啟動會詢問要不要從那裡繼續。
  - ⚠️ **Range… 會覆蓋範圍內所有舊片段，包括你手動標的片段**。整部影片只在**第一次**跑一次 Range…。
    之後如果還要預標，Start/End 只設在還沒檢查過的部分。如果跳出的警告寫著「tagged BY HAND」，請按 No。
  - 不要在 Range… 跑到一半時關閉程式：結果要整段跑完才會寫入，中途關閉的話，這一輪預標的結果就沒了（之前存好的標註不受影響）。
  - 片段是依**影片路徑**對應的，所以標到一半時**不要搬移或重新命名影片**，否則重新開啟時會看不到已經標好的片段。
- **標完一部影片就備份一次 `annotations.db`**（連同 `-wal`、`-shm` 檔，或者關掉程式後再複製）。

### 畫面配置

| 區域 | 用途 |
|---|---|
| 頂部工具列 | Open Video、Save、Export |
| 左側欄 → **Video Annotations** 分頁 | 選標籤（Service / In-Play / No-Play）、Mark Start → Mark End |
| 右側欄 → **Game State (Video)** | 設定模型路徑、按 **Range…** 用模型自動預標 |
| 底部 | 播放控制、依比賽狀態上色的進度條 |
| 最下方 **Temporal Timeline** | 每個標籤一列，可以拖曳、縮放片段 |

---

## 2. 標註步驟

### 2.1 開啟影片
頂部工具列 → **Open Video** → 選擇影片，例如 `D:\user\Downloads\2026_10_29 2-2_1080p60.mp4`。

### 2.2 先用模型預標（省時間）
1. 右側欄 **Game State (Video)** 列**預設已自動帶入** `src\weights\game_state_annotator.pt`，Ball / Players / Actions 也一樣，所以不用手動選。
   只有換成自己訓練的模型時，才需要按「…」選新的 `.pt`。選過一次會記在資料庫裡，之後就不會被預設值蓋掉。
   - 標註工具只吃單一 `.pt` 檔（`vmae_train.py` 的格式）。`src\weights\game_state\` 資料夾裡的
     `optimizer.pt`、`scheduler.pt` 是原作者訓練時留下的狀態，**不是模型**，選了會載入失敗。
   - 如果 `game_state_annotator.pt` 不存在，先從 HuggingFace 資料夾轉出來（在專案根目錄執行）：
     ```powershell
     .venv\Scripts\python.exe src\vb_gui\vb_annotator\train\hf_to_checkpoint.py
     ```
   - 之後自己訓練出來的 `.pt` 也是在這裡選。
2. 按 **Range…**：
   - Start frame / End frame 保持預設，就是整部影片。
   - **Window size 保持預設值（等於影片 fps，60fps 影片就是 60）**，也就是每次看 1 秒。
     **不要改成 30**，在 60fps 影片上只看 0.5 秒，幾乎全部會被判成 no-play。
3. 跑完後，Timeline 上會出現 `source="model"` 的片段，同一種狀態的連續片段會自動合併。

> 預標的品質不高：在這種機位下，大多數回合都會被漏掉。它的作用是給你一個起點，
> 主要工作還是在下一步，**把漏掉的回合補標成 In-Play / Service**。

### 2.3 逐回合檢查與修正

**操作鍵**

| 按鍵 | 功能 |
|---|---|
| `Space` | 播放 / 暫停 |
| `A` / `D`（或 ← / →） | 上一幀 / 下一幀 |
| `Q` / `E` | 往回 / 往前 15 幀（60fps 下是 0.25 秒） |
| `Shift+D` / `Shift+A` | 跳到下一個 / 上一個 play 片段（如果前面有 service，會停在 service 開頭） |
| `Alt+3` | 切換目前的影片標籤（Service → In-Play → No-Play） |
| `Ctrl+S` | 儲存，包括 Timeline 上還沒套用的拖曳修改 |

**補標一段（Mark Start → Mark End）**
1. 在左側欄 **Video Annotations** 選標籤，或按 `Alt+3` 切換。
2. 移到片段的第一幀，按 **Mark Start**，按鈕會變成 **Mark End**。
3. 往前移到片段的最後一幀，按 **Mark End**，這時才會寫入資料庫。
4. 標錯了就按 **Cancel**，整段放棄。

**修正模型預標的片段**
- 拖曳片段本體可以移動它，拖曳左右邊緣可以調整長度。改過的片段會顯示**虛線框**，還沒存檔。
- 按 **Apply Changes**（或 `Ctrl+S`）套用。想還原就按 **Cancel Changes**。
- 選取片段後按 **Delete Selected**，會**立即**刪除，沒辦法用 Cancel 復原。
- Timeline 標題列有縮放滑桿，放大後比較好對準邊界。

**建議的節奏**：先用 1 倍速播一遍，抓出每個回合的大概位置，再用 `Q`/`E`/`A`/`D` 修準邊界。

---

## 3. 標註規則（最重要）

> 完整規範與特殊情況請看 [`videomae_readme.md`](videomae_readme.md)。

**模型學的是你的標註，邊界定義前後一致，比單一邊界標得精準更重要。**

| 狀態 | 開始 | 結束 |
|---|---|---|
| **Service** | **擊球前約 1 秒**（發球員站在端線後、開始發球動作） | 球越過網子的那一幀 |
| **In-Play** | Service 結束的下一幀 | 球**落地、出界，或回合明顯結束**（死球）的那一幀 |
| **No-Play** | 上一個回合結束 | 下一個 Service 開始的前一幀 |

**原則**
- **每個回合都要有 Service**。Service 每段只有 1.2～3 秒，是最稀少的類別，漏標的話模型很難學會。拋球被遮擋時也要標（細節見 `docs/videomae_readme.md`）。
- 三種狀態**首尾相接**，不要留空白，也不要重疊。
- 撿球、走位、慶祝、暫停、換人、場邊閒聊，一律標 **No-Play**。
- 跳發的拋球和助跑都包含在 Service 裡（從擊球前約 1 秒開始算）。
- 看不清楚、判斷不了的片段（例如鏡頭被擋住），寧可整段標成 No-Play，或者乾脆不要用那部影片，**不要亂猜**。
- 熱身、練習球、沒有正式計分的回合：如果動作看起來和比賽一樣，就照比賽的規則標。

### 自我檢查清單（每部影片標完時做一次）
- [ ] 用 `Shift+D` 從頭跳一遍，每一個 play 的前面都有 service
- [ ] 回合數量和實際比分大致吻合（例如 25:20 加上幾個失誤，大概 45 個回合）
- [ ] 進度條上沒有明顯的空白或重疊
- [ ] 已經 `Ctrl+S`，並備份 `annotations.db`

### 需要標多少？

| 目標 | 建議量 |
|---|---|
| 能用 | 1 部影片完整標註（約 20～45 個回合）＋ 另 1 部當驗證 |
| 穩定 | 3～5 部同場地、同機位的影片 |

> **驗證集一定要用另一部影片**。如果同一回合切出來的片段同時出現在訓練集和驗證集，
> 準確率會虛高，等到換一部新影片才會發現模型其實不準。

一部 13 分鐘的影片，第一次標大概要 **30～45 分鐘**，熟練之後會更快。

---

## 4. 訓練與換上新模型

完整說明見 `src/game_state_trainer/README.md`。

```powershell
cd D:\coding\volleyball_analytics
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m game_state_trainer --list
.venv\Scripts\python.exe -m game_state_trainer --train "訓練影片.mp4" --val "驗證影片.mp4"
```

- 訓練和驗證**一定要用不同的影片**。
- 會從 `src/weights/game_state` 的主幹網路開始微調，分類層則重新初始化。原本的分類層權重異常，所以它的信心度永遠是 1.00。
- 開始訓練前會先評估原本的模型當作基準，訓練完會印出進步了多少。最好的模型存在 `output\game_state_models\<時間>\best`。
- 用新模型重跑剪輯工具：

  ```powershell
  .venv\Scripts\python.exe -m game_state_clipper "影片.mp4" --model output\game_state_models\<時間>\best
  ```

---

## 附錄：比賽狀態剪輯工具（`game_state_clipper`）

整部影片分類後，把非 no-play 的段落剪出來（13 分鐘的影片約 1.5 分鐘跑完）：

```powershell
cd D:\coding\volleyball_analytics
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m game_state_clipper "D:\user\Downloads\2026_10_29 2-2_1080p60.mp4"
```

輸出放在 `output\<影片名>_clips\`：剪好的 `clip_XXX_*.mp4`、`segments.csv`（各段起訖秒數）、`timeline.json`（每秒的判斷結果）。

| 參數 | 預設 | 說明 |
|---|---|---|
| `--window` | 1.0 | 每次分類涵蓋幾秒 |
| `--start` / `--end` | 整部 | 只跑某個區間（測試用） |
| `--smooth` | 3 | 多數決平滑，必須是奇數，1 = 關閉 |
| `--merge-gap` | 1.5 | 間隔小於此秒數的片段合併 |
| `--min-duration` | 1.0 | 短於此秒數的片段丟棄 |
| `--pad` | 1.0 | 每段前後各多留幾秒 |
| `--copy` | 關 | 不重新編碼，極快，但切點會對齊關鍵幀 |
| `--no-cut` | 關 | 只輸出 timeline / segments，不剪片 |
| `--model` | `src/weights/game_state` | VideoMAE 權重資料夾 |
