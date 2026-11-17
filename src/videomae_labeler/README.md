# VideoMAE Labeler（網頁版比賽狀態標註工具）

邊播放影片邊**按住**鍵盤標註 SERVICE / PLAY / NO-PLAY。按 Ctrl+S 才會寫進 vb_annotator 用的
`db/annotations.db`（`game_state_segments` 表，`end_frame` 包含最後一幀），所以原本的 VideoMAE 匯出也能直接使用。

**標註規範（什麼情況怎麼標）請看 [`docs/videomae_readme.md`](../../docs/videomae_readme.md)。**

## 啟動

```powershell
cd D:\coding\volleyball_analytics\src
..\.venv\Scripts\python.exe -m videomae_labeler          # 自動開啟 http://127.0.0.1:8765
..\.venv\Scripts\python.exe -m videomae_labeler --port 9000 --db D:\other\annotations.db --model D:\my_model --no-browser
```

也可以用環境變數 `VB_ANNOTATIONS_DB` 指定資料庫。

**模型**：右上角「模型預標」左邊的下拉選單會自動列出所有可用的模型，**預設選最新訓練出來的模型**：
- `output/game_state_models/<名稱>/best`：用 `game_state_trainer` 訓練出來的模型，選單上會顯示驗證集的 macro-F1
- 原始模型（`src/weights/game_state`，未微調）

選過的模型會被記住。訓練出新模型後，重新整理網頁就會出現在選單裡。`--model <資料夾>` 可以額外加入一個模型，並設成預設。

按「開啟影片」會跳出 Windows 的選擇檔案視窗，預設從 `D:/user/Downloads` 開始；之後會記住上次開啟的資料夾。
要換預設資料夾，可以用 `--video-dir` 或環境變數 `VB_VIDEO_DIR`。
「已標註的影片」會列出資料庫中已經有標註的影片，也可以直接貼上影片路徑。

> 為什麼不用瀏覽器內建的選擇檔案？因為瀏覽器拿不到檔案的完整路徑，而標註是用影片的完整路徑對應的。

## 運作方式

- **模型預標要手動按**：選好模型後按右上角的「**模型預標**」，才會在背景用 VideoMAE 預標整部影片（有 GPU 時會用 CUDA），開啟影片時不會自動執行。
  模型判成 play / service 的保留，**其餘全部補成 no-play**。預標期間就可以開始標註，模型跑完後會合併進來，而且不會蓋掉你標的部分。
- **按住 `1` / `2` / `3` 才會標註**：影片播放時，按住期間顯示過的每一幀都標成該狀態，放開就停止。
  同時按住兩個鍵時，以最後按下的為準。暫停時按住不會標註，按住後再開始播放就會開始標。
- **和模型重疊時，以你按的為準**。存檔前，相鄰且狀態相同的片段會合併成一段：只要有任何一部分是手動標的就記為 `manual`，`confidence` 是各段依長度加權的平均。
- `Delete` 刪除選取的片段後，那整段會變回預設 no-play。
- **不會自動存檔**：所有修改都只存在網頁裡，按 `Ctrl+S`（或「儲存」按鈕）才會寫進資料庫。
  有未儲存的變更時，右上角會顯示提示，關閉網頁或切換影片前也會先詢問你。

時間軸上的顏色：實心是手動標註，中等透明度是模型預測，最淡的是預設 no-play。

| 快捷鍵 | 功能 |
|---|---|
| `Space` | 播放 / 暫停 |
| 按住 `1` `2` `3` | 標註 Service / Play / No-play |
| `A` `←` ／ `D` `→` | 倒退 / 快轉 3 秒 |
| `,` `.` | 上一幀 / 下一幀 |
| `Home` 或 ⏮ 按鈕 | 回到 0:00 |
| `[` `]` | 播放速度 |
| `Delete` | 刪除選取的片段 |
| `Ctrl+Z` ／ `Ctrl+Y` | 復原 / 重做 |
| `Ctrl+S` | 儲存到資料庫 |
| `Esc` | 取消選取 |

點選時間軸上的片段後，可以在右側修改狀態、起點和終點，或直接拖曳片段邊緣。

## 注意

- 存檔時會把**整部影片**的時間軸都存下來（手動標註是 `manual`；模型預測和預設 no-play 是 `model`，預設 no-play 的 confidence 是 0）。
- 瀏覽器無法播放的編碼（例如 OpenCV 輸出的 MPEG-4 Part 2）會自動用 ffmpeg 轉成 H.264 相容檔，存在資料庫旁邊的 `labeler_proxies/`。轉出來的檔案幀號和原影片一樣。
- 不要同時用 vb_annotator 和本工具編輯**同一部**影片，後存的會覆蓋先存的。
- 伺服器只綁 `127.0.0.1`，因為 API 可以讀取本機的影片檔。

## 測試

```powershell
.venv\Scripts\python.exe -m pytest tests\videomae_labeler
node --test tests\videomae_labeler\recorder.test.mjs tests\videomae_labeler\ops.test.mjs
```
