"""data.py  -  TV1: Dữ liệu và augmentation cho FER2013 (PyTorch).

API chung của nhóm (không đổi tên để code song song):
    get_loaders(data_path, aug='full', batch_size=128, ...) -> (train_loader, val_loader, test_loader)
    compute_class_weights(labels) -> torch.FloatTensor (dùng cho CrossEntropyLoss(weight=...))

Quy ước split:  Training -> train,  PublicTest -> val (chọn checkpoint),  PrivateTest -> test (báo cáo cuối).
Thứ tự lớp (cố định cho cả nhóm, theo thứ tự chữ cái như code mẫu):
    0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Neutral, 5 Sad, 6 Surprise

Chạy riêng để sinh hình cho report:
    python data.py --data fer2013.tar.gz --out figs/
"""
import argparse
import glob
import os
import tarfile
import zipfile

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# Nhãn trong fer2013.csv: 0=Angry 1=Disgust 2=Fear 3=Happy 4=Sad 5=Surprise 6=Neutral
# -> đổi sang thứ tự chữ cái ở trên.
_CSV_TO_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 4}
_USAGE_TO_SPLIT = {'Training': 'train', 'PublicTest': 'val', 'PrivateTest': 'test'}
_EXPECTED_SIZES = {'train': 28709, 'val': 3589, 'test': 3589}

# 4 mức augmentation cho thí nghiệm của TV1 (mức thứ 5 là 'full' + class weight)
AUG_MODES = ('none', 'flip', 'flip_rot', 'full')


# ----------------------------------------------------------------------------
# 1. Giải nén + đọc dữ liệu
# ----------------------------------------------------------------------------
def _find_csv(path, work_dir):
    """Nhận .csv, thư mục, .tar.gz hoặc .zip và trả về đường dẫn fer2013.csv."""
    if path.endswith('.csv') and os.path.isfile(path):
        return path
    if os.path.isfile(path):
        os.makedirs(work_dir, exist_ok=True)
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as tf:
                tf.extractall(work_dir)
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                zf.extractall(work_dir)
        else:
            raise ValueError('Không nhận ra định dạng file: %s' % path)
        path = work_dir
    for name in ('fer2013.csv', 'icml_face_data.csv'):      # 2 file này chứa cùng dữ liệu
        hits = glob.glob(os.path.join(path, '**', name), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError('Không tìm thấy fer2013.csv hoặc icml_face_data.csv trong %s' % path)


def load_fer2013(data_path, cache='fer2013_cache.npz', work_dir='fer2013_extracted'):
    """Trả về dict {'train': (x, y), 'val': (x, y), 'test': (x, y)}.
    x: uint8 (N, 48, 48); y: int64 (N,) theo thứ tự CLASSES. Có cache .npz để lần sau đọc nhanh."""
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return {s: (d[s + '_x'], d[s + '_y']) for s in _EXPECTED_SIZES}

    import pandas as pd
    df = pd.read_csv(_find_csv(data_path, work_dir))
    df.columns = [c.strip() for c in df.columns]            # icml_face_data.csv có tên cột ' Usage', ' pixels'
    if not {'emotion', 'pixels', 'Usage'} <= set(df.columns):
        raise ValueError('File phải có đủ 3 cột emotion, pixels, Usage (không dùng train.csv/test.csv). Cột hiện có: %s' % list(df.columns))
    x = np.stack([np.array(p.split(), dtype=np.uint8) for p in df['pixels']]).reshape(-1, 48, 48)
    y = df['emotion'].map(_CSV_TO_IDX).to_numpy(dtype=np.int64)

    out = {}
    for usage, split in _USAGE_TO_SPLIT.items():
        m = (df['Usage'] == usage).to_numpy()
        out[split] = (x[m], y[m])
    if cache:
        np.savez_compressed(cache, **{s + '_x': out[s][0] for s in out},
                            **{s + '_y': out[s][1] for s in out})
    return out


# ----------------------------------------------------------------------------
# 2. Dataset + augmentation
# ----------------------------------------------------------------------------
def build_transforms(aug, mean=0.5, std=0.25, normalize=True):
    """aug: 'none' | 'flip' | 'flip_rot' | 'full'.
    'full' = lật ngang + xoay 10 độ + dịch 10% + zoom 10% (giống code mẫu Keras)."""
    if aug not in AUG_MODES:
        raise ValueError('aug phải thuộc %s' % (AUG_MODES,))
    ops = []
    if aug in ('flip', 'flip_rot', 'full'):
        ops.append(T.RandomHorizontalFlip(p=0.5))
    if aug == 'flip_rot':
        ops.append(T.RandomRotation(10))
    if aug == 'full':
        ops.append(T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    ops.append(T.ToTensor())                      # [0,255] -> [0,1], shape (1,48,48)
    if normalize:
        ops.append(T.Normalize([mean], [std]))
    return T.Compose(ops)


class FERDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x, self.y, self.transform = x, y, transform
        self.labels = y                            # để tính class weight / thống kê

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = Image.fromarray(self.x[i])           # uint8 2D -> ảnh xám 'L'
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[i])


def compute_class_weights(labels, n_classes=7):
    """Trọng số lớp = căn bậc hai của trọng số nghịch tần suất: w_c = sqrt(N / (K * n_c)).
    Lớp Disgust (436 ảnh) có trọng số lớn nhất (khoảng 3.1; bản không căn là khoảng 9.4 làm mạng không hội tụ, xem nhật ký A5)."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(np.sqrt(w), dtype=torch.float32)


def get_loaders(data_path, aug='full', batch_size=128, num_workers=2, seed=42,
                cache='fer2013_cache.npz'):
    """Trả về (train_loader, val_loader, test_loader).
    Chỉ tập train được augment; val/test chỉ chuẩn hoá. mean/std tính trên tập train."""
    data = load_fer2013(data_path, cache=cache)
    for s, n in _EXPECTED_SIZES.items():
        if len(data[s][0]) != n:
            print('[cảnh báo] split %s có %d ảnh, kỳ vọng %d' % (s, len(data[s][0]), n))

    xtr = data['train'][0].astype(np.float32) / 255.0
    mean, std = float(xtr.mean()), float(xtr.std())

    tf_train = build_transforms(aug, mean, std)
    tf_eval = build_transforms('none', mean, std)
    ds_train = FERDataset(*data['train'], transform=tf_train)
    ds_val = FERDataset(*data['val'], transform=tf_eval)
    ds_test = FERDataset(*data['test'], transform=tf_eval)
    for ds in (ds_train, ds_val, ds_test):
        ds.mean, ds.std = mean, std

    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=g,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------------
# 3. Hình cho report: histogram lớp, ảnh mẫu, ảnh sau augmentation
# ----------------------------------------------------------------------------
def plot_class_histogram(data, save_path):
    import matplotlib.pyplot as plt
    names = {'train': 'Training', 'val': 'PublicTest (val)', 'test': 'PrivateTest (test)'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, s in zip(axes, ('train', 'val', 'test')):
        counts = np.bincount(data[s][1], minlength=len(CLASSES))
        bars = ax.bar(CLASSES, counts)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha='center', va='bottom', fontsize=9)
        ax.set_title('%s - %d ảnh' % (names[s], counts.sum()))
        ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_samples(x, y, save_path, rows=4, cols=8, seed=0):
    import matplotlib.pyplot as plt
    idx = np.random.RandomState(seed).choice(len(x), rows * cols, replace=False)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(x[i], cmap='gray')
        ax.set_title(CLASSES[y[i]], fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_aug_examples(x, y, save_path, n=6, seed=0):
    """Mỗi hàng = 1 mức augmentation, cột đầu là ảnh gốc."""
    import matplotlib.pyplot as plt
    idx = np.random.RandomState(seed).choice(len(x), n, replace=False)
    fig, axes = plt.subplots(len(AUG_MODES), n, figsize=(2 * n, 2.2 * len(AUG_MODES)))
    for r, mode in enumerate(AUG_MODES):
        tf = build_transforms(mode, normalize=False)
        for c, i in enumerate(idx):
            torch.manual_seed(seed + c)
            img = tf(Image.fromarray(x[i])).squeeze(0).numpy()
            axes[r, c].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[r, c].axis('off')
            if c == 0:
                axes[r, c].set_title(mode, fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='fer2013.csv / .tar.gz / .zip / thư mục chứa fer2013.csv')
    ap.add_argument('--out', default='figs')
    ap.add_argument('--cache', default='fer2013_cache.npz')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data = load_fer2013(a.data, cache=a.cache)
    for s in ('train', 'val', 'test'):
        print('%-5s %6d ảnh, shape %s, đếm lớp: %s' % (
            s, len(data[s][0]), data[s][0].shape[1:], np.bincount(data[s][1], minlength=7).tolist()))
    print('Class weight (train):', compute_class_weights(data['train'][1]).numpy().round(3).tolist())

    plot_class_histogram(data, os.path.join(a.out, 'histogram_classes.png'))
    plot_samples(*data['train'], os.path.join(a.out, 'sample_images.png'))
    plot_aug_examples(*data['train'], os.path.join(a.out, 'augmentation_examples.png'))
    print('Đã lưu hình vào', a.out)


if __name__ == '__main__':
    main()
