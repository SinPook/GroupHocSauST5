# BTLON02 - Nhận dạng cảm xúc khuôn mặt FER2013 (PyTorch)

Nhóm 4 thành viên. Mỗi người một module code, 5 lần chạy thí nghiệm, một phần report và slide.

| Thành viên | File code | Thí nghiệm (mã trong nhật ký) |
|---|---|---|
| TV1 (nhóm trưởng) | `data.py` (+ `run_aug_experiments.py`) | A1 đến A5: augmentation, class weight |
| TV2 | `models.py` | B1 đến B5: kiến trúc, BatchNorm, activation |
| TV3 | `train.py` | C1 đến C5: optimizer, LR, batch size |
| TV4 | `evaluate.py` | D1 đến D5: transfer learning, TTA, ensemble |

## 1. Quy ước chung (cả nhóm phải theo đúng)

- **Baseline chung:** Medium CNN, Adam lr 1e-3, batch 128, augmentation đầy đủ (lật ngang + xoay 10° + dịch 10% + zoom 10%). Mỗi thí nghiệm chỉ đổi 1 yếu tố so với baseline.
- **Chia dữ liệu:** `Training` -> train, `PublicTest` -> val (chọn checkpoint), `PrivateTest` -> test (chỉ báo cáo kết quả cuối, đo 1 lần).
- **Thứ tự lớp (cố định):** 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Neutral, 5 Sad, 6 Surprise (thứ tự chữ cái như code mẫu; khác thứ tự nhãn gốc trong `fer2013.csv`, `data.py` đã đổi sẵn).
- **Đầu vào:** ảnh xám 48x48, tensor `(1, 48, 48)`, chuẩn hoá bằng mean/std tính trên tập train.
- **Chọn checkpoint:** theo `val_loss` nhỏ nhất trên PublicTest. Early stopping patience 10, tối đa 100 epoch.
- **Seed mặc định:** 42.

## 2. Chữ ký hàm (hợp đồng chung)

```python
# data.py  (TV1)
get_loaders(data_path, aug='full', batch_size=128, num_workers=2, seed=42,
            cache='fer2013_cache.npz') -> (train_loader, val_loader, test_loader)
    # aug: 'none' | 'flip' | 'flip_rot' | 'full'
compute_class_weights(labels, n_classes=7) -> torch.FloatTensor
    # căn bậc hai của trọng số nghịch tần suất, dùng cho CrossEntropyLoss(weight=...)

# models.py  (TV2)
build_model(name='medium', activation='relu', batchnorm=False) -> nn.Module
    # name: 'small' | 'medium' | 'large'; activation: 'relu' | 'leakyrelu'

# train.py  (TV3)
train(model, train_loader, val_loader, epochs, lr, optimizer, class_weights,
      patience, ckpt_path, device) -> dict(best_epoch, val_loss, val_acc, history)
    # history: list[dict(epoch, train_loss, train_acc, val_loss, val_acc)]
evaluate(model, loader, device, class_weights=None) -> dict(loss, acc)

# evaluate.py  (TV4)
    # chạy trên checkpoint tốt nhất: PrivateTest, confusion matrix, precision/recall/F1,
    # ảnh đoán đúng/sai, filter conv cuối và feature map
```

Thứ tự phụ thuộc: `data.py` -> `models.py` -> `train.py` -> chạy thí nghiệm -> `evaluate.py` trên checkpoint tốt nhất.

## 3. Lưu ý để A4 trùng với B2 (baseline chung)

Các lần chạy A1 đến A5 của TV1 đã chạy bằng **bản dự phòng** nằm cuối `run_aug_experiments.py` (script in `dùng code của nhóm: False`). Để kết quả A4 dùng được làm dòng B2 của TV2, `models.py` và `train.py` phải **đúng như bản dự phòng**:

- **Medium CNN:** 2 conv 64, 3 conv 128, 3 conv 256 (kernel 3x3, padding 1, ReLU); sau mỗi khối MaxPool(2) + Dropout(0.2); Flatten -> Linear(256*6*6, 1024) -> ReLU -> Dropout(0.5) -> Linear(1024, 7). Không BatchNorm. Trả về logits (không softmax).
- **Vòng train:** Adam lr 1e-3, `CrossEntropyLoss`, lưu checkpoint khi `val_loss` giảm, dừng sau 10 epoch không cải thiện, val loss tính không dùng class weight.

Nếu bản của TV2 hoặc TV3 khác những điểm trên thì A4 và B2 là hai thí nghiệm khác nhau: khi đó TV2 phải tự chạy lại B2. Khi `models.py` và `train.py` có đủ chữ ký ở mục 2, `run_aug_experiments.py` tự dùng code của nhóm (không cần sửa).

## 4. Cách chạy

```bash
# Dữ liệu: fer2013.csv (cột emotion, pixels, Usage) hoặc .tar.gz / .zip chứa file đó
python data.py --data fer2013.csv --out figs/            # hình histogram, ảnh mẫu, ảnh augmentation
python run_aug_experiments.py --data fer2013.csv --out runs_tv1 --log NhatKy_ThucNghiem.xlsx
python run_aug_experiments.py --data fer2013.csv --only A5 --seed 7 --log NhatKy_ThucNghiem.xlsx   # chạy lại vài mã
```

Yêu cầu: Python 3, PyTorch, torchvision, numpy, pandas, Pillow, matplotlib, openpyxl; nên dùng GPU (Colab T4: mỗi lần chạy khoảng 5 đến 20 phút).

## 5. Kết quả TV1 (PublicTest = val, PrivateTest = test)

| Mã | Augmentation | Class weight | Best epoch | val_loss | val_acc | test_acc |
|---|---|---|---|---|---|---|
| A1 | không | không | 13 | 1.0690 | 61.05% | 62.22% |
| A2 | lật ngang | không | 15 | 1.0297 | 61.44% | 63.08% |
| A3 | lật + xoay 10° | không | 38 | 0.9882 | 64.03% | 64.47% |
| A4 | đầy đủ (baseline) | không | 56 | 0.9689 | 63.95% | 65.00% |
| A5 | đầy đủ | có (căn bậc hai, seed 7) | 44 | 0.9877 | 62.66% | 63.42% |

Ghi chú: A5 lần đầu dùng trọng số gốc (Disgust khoảng 9.4) nên mạng không học (loss kẹt ở ln 7 = 1.946, val_acc 24.9% = tỉ lệ lớp Happy), vì vậy chuyển sang căn bậc hai. Mỗi cấu hình chỉ chạy 1 seed nên chênh lệch dưới 1 điểm nằm trong mức nhiễu.
