# Court36 相機方位與點序：Blender 專用 Codex Prompt

請在 `path\to\volley-court-training\blender` 內實作，保留現有 36 點定義與可見度語意。

## 這個專案的判斷方式

Blender 已知真實 `camera.location = (x, y, z)`，不要從渲染圖片或 homography 猜視角。

球場座標以 `volleyball_synthetic/constants.py` 為準：

- `x: 0 → 9`：左到右
- `y: 0 → 18`：遠端到近端
- `z`：高度

先把實體鏡頭分成 8 方位，再只合併 180° 對稱，得到 4 種訓練點序：

| 8 方位來源 | 4 種輸出點序 |
|---|---|
| `FAR_CENTER`, `NEAR_CENTER` | `ENDLINE` |
| `LEFT_SIDE`, `RIGHT_SIDE` | `SIDELINE` |
| `FAR_LEFT`, `NEAR_RIGHT` | `DIAGONAL_BOTTOM_RIGHT` (`\\`) |
| `FAR_RIGHT`, `NEAR_LEFT` | `DIAGONAL_TOP_RIGHT` (`/`) |

不要把兩種斜角合成一種；它們是鏡像，不是 180° 對稱。

## 請修改的位置

- `volleyball_synthetic/randomization.py`
  - 在 `_sample_camera_location()` 取得座標後，依 `x/y` 判定 8 方位。
  - `corner` 可直接得到四個角；`endline` 得到遠／近；各種 `*_sideline` 得到左／右。
  - `overhead` 或位於球場內的鏡頭也必須歸入四模式：改看投影後球場長軸，並固定「畫面越上方越遠」。直向=`ENDLINE`、橫向=`SIDELINE`、兩種斜率分別對應兩個 diagonal。
- `volleyball_synthetic/labels.py`
  - 寫 YOLO Pose 前套用固定的 36-index permutation。
  - 只能交換 index；每點的 `(x, y, visibility)` 必須一起移動，不能重新估座標。
- `volleyball_synthetic/dataset.py`
  - metadata 增加 `source_camera_position`、`canonical_camera_mode`、`keypoint_permutation`。
  - 保留原本 `camera.location`，讓輸出可追溯。
- `volleyball_synthetic/constants.py`
  - 36 個 `KEYPOINT_NAMES` 與 `KEYPOINT_WORLD` 不得增刪或改幾何定義。

## 驗收

1. 寫測試覆蓋完整 8 → 4 對應。
2. 驗證轉換前後仍為 36 點，visibility 與座標只隨 index 搬移。
3. 用 `visualize_yolo_pose.py` 抽查四種模式；同一模式的 `KP00` 必須固定代表同一個球場端點。
4. 只要可投影出球場，就必須輸出四模式之一；無法投影時才回報錯誤，不要靜默輸出混亂點序。

參考圖：

- [8 方位合併成 4 模式](docs/court36-camera-modes/camera-8-to-4.svg)
- [正確的 36 點身分](docs/court36-camera-modes/reference-correct-36-points.png)
- [錯誤：混用點序後的推論](docs/court36-camera-modes/error-model-mixed-order.png)
