# BTL01 - Function Approximation

## Cấu trúc
```
BTL01_Nhom_HoTen_MSSV/
├── data_gaussian.py   # [TV1:Nguyễn Lê Tấn Phúc] có sẵn: sinh/chia/kiểm tra/vẽ dữ liệu, get_data(), BASELINE
├── model.py           # [TV2] MLP (nn.Module)           
├── train.py           # [TV2] vòng lặp train/validation 
├── evaluate.py        # [TV2] MSE, MAE/RMSE, hình y–ŷ   
├── run_A.py           # [TV3] thí nghiệm A (số mẫu)     
├── run_B.py           # [TV3] thí nghiệm B (số nơ-ron)  
├── run_C.py           # [TV4] thí nghiệm C (số tầng ẩn) 
├── requirements.txt
├── README.md          # [TV1:Nguyễn Lê Tấn Phúc]
├── data/              # train_pool.csv, val.csv, test.csv 
├── figures/           # hình dữ liệu 
├── results/           # data_statistics.csv (thêm log CSV các thí nghiệm)
├── checkpoints/       # trọng số .pt của mô hình được chọn
├── report/            # BaoCao
└── slides/            # Slide
```
Chạy mọi lệnh từ thư mục gốc, ví dụ `python run_A.py`.

## Cài đặt và chạy phần dữ liệu
```bash
pip install -r requirements.txt     # thêm torch khi chạy phần mô hình
python data_gaussian.py             # sinh CSV, kiểm tra, vẽ hình, HIỆN cửa sổ ảnh
python data_gaussian.py --no-show   # chỉ lưu ảnh vào figures/, không hiện cửa sổ
```
Mặc định chương trình vừa lưu ảnh vừa hiện cửa sổ (`SHOW_FIGURES = True` trong file). Đóng các cửa sổ ảnh thì chương trình mới kết thúc. Trên máy không có màn hình, dùng `--no-show`.

## Dữ liệu (cố định cho cả nhóm)
- f(x) = 1/(2π√det C) · exp(−½ (x−μ)ᵀ C⁻¹ (x−μ)), μ = (0, 0), C = [[0.3, −0.5], [−0.5, 2]].
- x₁ ~ U(−2, 2), x₂ ~ U(−4, 4), y = f(x₁, x₂).

| Tập | Số mẫu | Seed | Ghi chú |
|---|---|---|---|
| Train pool | 2000 | 42 | Thí nghiệm A lấy n mẫu **đầu tiên** (tập con lồng nhau) |
| Validation | 400 | 43 | Chọn mô hình, checkpoint, early stopping |
| Test | 100 | 44 | Chỉ đánh giá cuối, không dùng chọn siêu tham số |

Không đổi seed/kích thước tập trong `data_gaussian.py`. Seed của thí nghiệm (`BASELINE["seed"]`) chỉ đổi khởi tạo trọng số và thứ tự shuffle.

## Dùng dữ liệu trong code
```python
from data_gaussian import get_data, BASELINE
d = get_data(n_train=600, as_torch=True)
# d["X_train"], d["y_train"] (N,1), d["X_val"], d["y_val"], d["X_test"], d["y_test"]
```

