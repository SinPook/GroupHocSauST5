"""run_aug_experiments.py - TV1: chạy 5 thí nghiệm augmentation trên baseline Medium CNN.

Baseline chung: Medium CNN, Adam lr 1e-3, batch 128. Mỗi lần chỉ đổi augmentation (hoặc class weight):
    A1 none | A2 flip | A3 flip_rot | A4 full (= baseline) | A5 full + class weight

Cách chạy (Colab / máy có GPU):
    python run_aug_experiments.py --data fer2013.tar.gz --out runs_tv1
    python run_aug_experiments.py --data fer2013.tar.gz --only A1 A5        # chạy lại vài lần
    python run_aug_experiments.py --data fer2013.tar.gz --log NhatKy_ThucNghiem.xlsx   # tự điền nhật ký

HỢP ĐỒNG HÀM CHUNG (nhóm cần thống nhất; nếu models.py/train.py của TV2/TV3 đúng chữ ký này thì script
tự dùng, nếu chưa có thì dùng bản dự phòng ở cuối file để TV1 chạy trước):
    models.build_model(name='medium', activation='relu', batchnorm=False) -> nn.Module
    train.train(model, train_loader, val_loader, epochs, lr, optimizer, class_weights, patience,
                ckpt_path, device) -> dict(best_epoch, val_loss, val_acc, history)   # history: list[dict]
    train.evaluate(model, loader, device, class_weights=None) -> dict(loss, acc)
Checkpoint được chọn theo val_loss trên PublicTest; test_acc đo 1 lần trên PrivateTest.
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data import get_loaders, compute_class_weights, load_fer2013

# (mã, augmentation, dùng class weight, mô tả)
EXPERIMENTS = [
    ('A1', 'none',     False, 'Không augment'),
    ('A2', 'flip',     False, 'Chỉ lật ngang'),
    ('A3', 'flip_rot', False, 'Lật + xoay 10'),
    ('A4', 'full',     False, 'Lật + xoay + dịch/zoom (= baseline)'),
    ('A5', 'full',     True,  'Baseline + class weight'),
]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Lấy hàm của nhóm nếu có, không thì dùng bản dự phòng bên dưới
# ----------------------------------------------------------------------------
try:
    from models import build_model as _team_build
    from train import train as _team_train, evaluate as _team_eval
    USING_TEAM_CODE = True
except ImportError:
    USING_TEAM_CODE = False


def build_model(name='medium'):
    return _team_build(name=name, activation='relu', batchnorm=False) if USING_TEAM_CODE else FallbackMediumCNN()


def run_training(model, tr, va, args, class_weights, ckpt):
    fn = _team_train if USING_TEAM_CODE else fallback_train
    return fn(model, tr, va, epochs=args.epochs, lr=1e-3, optimizer='adam', class_weights=class_weights,
              patience=args.patience, ckpt_path=ckpt, device=args.device)


def run_eval(model, loader, args):
    fn = _team_eval if USING_TEAM_CODE else fallback_evaluate
    return fn(model, loader, args.device)


# ----------------------------------------------------------------------------
# Chạy 1 thí nghiệm
# ----------------------------------------------------------------------------
def run_one(code, aug, use_cw, desc, args):
    print('\n===== %s: %s (aug=%s, class_weight=%s) =====' % (code, desc, aug, use_cw))
    set_seed(args.seed)
    tr, va, te = get_loaders(args.data, aug=aug, batch_size=128, num_workers=args.workers, seed=args.seed)
    cw = compute_class_weights(tr.dataset.labels).to(args.device) if use_cw else None

    model = build_model('medium').to(args.device)
    ckpt = os.path.join(args.out, '%s_best.pt' % code)
    t0 = time.time()
    res = run_training(model, tr, va, args, cw, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=args.device))   # nạp checkpoint tốt nhất (theo val)
    val = run_eval(model, va, args)
    test = run_eval(model, te, args)                                      # PrivateTest: chỉ đo 1 lần
    row = dict(code=code, aug=aug, class_weight='Có' if use_cw else 'Không',
               best_epoch=res['best_epoch'], val_loss=round(val['loss'], 4), val_acc=round(val['acc'], 4),
               test_acc=round(test['acc'], 4), minutes=round((time.time() - t0) / 60, 1))
    print('->', row)
    save_curves(res['history'], os.path.join(args.out, '%s_curves.png' % code), code, desc)
    return row


def save_curves(history, path, code, desc):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ep = [h['epoch'] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h['train_acc'] for h in history], label='train'); ax[0].plot(ep, [h['val_acc'] for h in history], label='val')
    ax[0].set_title('Accuracy'); ax[0].set_xlabel('epoch'); ax[0].legend()
    ax[1].plot(ep, [h['train_loss'] for h in history], label='train'); ax[1].plot(ep, [h['val_loss'] for h in history], label='val')
    ax[1].set_title('Loss'); ax[1].set_xlabel('epoch'); ax[1].legend()
    fig.suptitle('%s - %s' % (code, desc)); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def append_csv(rows, path):
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def update_log_xlsx(rows, xlsx_path):
    """Điền Best epoch, val_loss, val_acc, test_acc, ngày chạy vào đúng dòng (theo cột Mã) trong NhatKy."""
    from datetime import date
    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path); ws = wb['NhatKy']
    where = {ws.cell(r, 1).value: r for r in range(2, ws.max_row + 1)}
    for row in rows:
        r = where.get(row['code'])
        if r is None:
            print('[cảnh báo] không thấy mã %s trong nhật ký' % row['code']); continue
        ws.cell(r, 11, row['best_epoch']); ws.cell(r, 12, row['val_loss'])
        ws.cell(r, 13, row['val_acc']);    ws.cell(r, 14, row['test_acc'])
        ws.cell(r, 16, date.today())
    wb.save(xlsx_path)
    print('Đã cập nhật nhật ký:', xlsx_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='runs_tv1')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--only', nargs='*', help='chỉ chạy các mã này, vd A1 A5')
    ap.add_argument('--log', help='đường dẫn NhatKy_ThucNghiem.xlsx để tự điền kết quả')
    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    print('Thiết bị:', args.device, '| dùng code của nhóm:' , USING_TEAM_CODE)
    load_fer2013(args.data)                                    # tạo cache 1 lần

    rows = []
    for code, aug, cw, desc in EXPERIMENTS:
        if args.only and code not in args.only:
            continue
        rows.append(run_one(code, aug, cw, desc, args))
        append_csv(rows[-1:], os.path.join(args.out, 'results_tv1.csv'))   # ghi ngay, lỡ ngắt giữa chừng
    if rows and args.log:
        update_log_xlsx(rows, args.log)
    print('\nXong. Kết quả:', os.path.join(args.out, 'results_tv1.csv'))


# ----------------------------------------------------------------------------
# Bản dự phòng (chỉ dùng khi chưa có models.py / train.py của TV2 / TV3)
# ----------------------------------------------------------------------------
class FallbackMediumCNN(nn.Module):
    """Medium CNN như code mẫu: 2x64, 3x128, 3x256 conv (3x3), mỗi khối MaxPool + Dropout 0.2; FC 1024 + Dropout 0.5."""
    def __init__(self, n_classes=7):
        super().__init__()
        def block(cin, cout, n):
            layers = []
            for i in range(n):
                layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1), nn.ReLU(inplace=True)]
            return layers + [nn.MaxPool2d(2), nn.Dropout(0.2)]
        self.features = nn.Sequential(*block(1, 64, 2), *block(64, 128, 3), *block(128, 256, 3))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256 * 6 * 6, 1024), nn.ReLU(inplace=True),
                                        nn.Dropout(0.5), nn.Linear(1024, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def fallback_evaluate(model, loader, device, class_weights=None):
    model.eval(); crit = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    loss = correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += crit(out, y).item(); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return {'loss': loss / n, 'acc': correct / n}


def fallback_train(model, train_loader, val_loader, epochs, lr, optimizer, class_weights, patience, ckpt_path, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    best, best_epoch, bad, history = float('inf'), 0, 0, []
    for ep in range(1, epochs + 1):
        model.train(); tl = tc = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); out = model(x); loss = crit(out, y); loss.backward(); opt.step()
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
        v = fallback_evaluate(model, val_loader, device)               # val_loss không dùng class weight
        history.append(dict(epoch=ep, train_loss=tl / n, train_acc=tc / n, val_loss=v['loss'], val_acc=v['acc']))
        print('ep %3d  train %.3f/%.3f  val %.3f/%.3f' % (ep, tl / n, tc / n, v['loss'], v['acc']))
        if v['loss'] < best:
            best, best_epoch, bad = v['loss'], ep, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print('Early stopping'); break
    return dict(best_epoch=best_epoch, val_loss=best, val_acc=history[best_epoch - 1]['val_acc'], history=history)


if __name__ == '__main__':
    main()
