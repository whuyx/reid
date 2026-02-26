import sys
import os
import heapq
import numpy as np
from PIL import Image
import torch
import torchreid
from torchvision import transforms
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# =========================
# 异步检索线程
# =========================
class SearchThread(QThread):
    finished = pyqtSignal(list)

    def __init__(self, model, transform, gallery_paths, query_path, device):
        super().__init__()
        self.model = model
        self.transform = transform
        self.gallery_paths = gallery_paths
        self.query_path = query_path
        self.device = device

    def extract_feature(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img)
        return feat.cpu().numpy().flatten()

    def run(self):
        # 先提取 query 特征
        query_feat = self.extract_feature(self.query_path)

        # 读取 gallery 特征，如果没有缓存就生成
        feat_cache_path = "gallery_feats.npy"
        if os.path.exists(feat_cache_path):
            gallery_feats = np.load(feat_cache_path)
        else:
            gallery_feats = []
            for g_path in self.gallery_paths:
                g_feat = self.extract_feature(g_path)
                gallery_feats.append(g_feat)
            gallery_feats = np.array(gallery_feats)
            np.save(feat_cache_path, gallery_feats)

        # 计算距离并取 Top-10
        dists = np.sum((gallery_feats - query_feat)**2, axis=1)
        topk_idx = np.argsort(dists)[:10]
        topk_paths = [self.gallery_paths[i] for i in topk_idx]

        self.finished.emit(topk_paths)


# =========================
# 主 GUI
# =========================
class ReIDApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("行人重识别系统 Demo")
        self.resize(1100, 500)

        self.init_ui()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ==========================
        # 加载模型
        # ==========================
        self.model = torchreid.models.build_model(
            name='resnet50',
            num_classes=751,   # Market1501训练时的ID数
            loss='softmax',
            pretrained=False
        )
        checkpoint_path = "E:/pylearning/reid_project/logs/resnet50/model/model.pth.tar-60"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # ==========================
        # 数据预处理
        # ==========================
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # ==========================
        # 加载图库路径
        # ==========================
        self.gallery_dir = "E:/pylearning/reid_data/market1501/bounding_box_test"
        self.gallery_paths = [os.path.join(self.gallery_dir, f)
                              for f in os.listdir(self.gallery_dir) if f.endswith('.jpg')]

    # ==========================
    # GUI 初始化
    # ==========================
    def init_ui(self):
        # 左侧：查询图
        self.query_label = QLabel("Query Image")
        self.query_label.setAlignment(Qt.AlignCenter)
        self.query_label.setFixedSize(200, 300)
        self.query_label.setStyleSheet("border:1px solid gray")

        self.btn_load = QPushButton("选择图片")
        self.btn_search = QPushButton("开始检索")

        self.btn_load.clicked.connect(self.load_image)
        self.btn_search.clicked.connect(self.start_search)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.query_label)
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.btn_search)
        left_layout.addStretch()

        # 右侧：Top-K
        self.result_labels = []
        right_layout = QHBoxLayout()
        for i in range(10):
            label = QLabel(f"Top-{i+1}")
            label.setFixedSize(120, 180)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border:1px solid gray")
            self.result_labels.append(label)
            right_layout.addWidget(label)

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    # ==========================
    # 选择图片
    # ==========================
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择行人图片", "", "Image Files (*.jpg *.png)"
        )
        if path:
            pixmap = QPixmap(path).scaled(
                self.query_label.size(), Qt.KeepAspectRatio
            )
            self.query_label.setPixmap(pixmap)
            self.query_path = path

    # ==========================
    # 点击检索按钮
    # ==========================
    def start_search(self):
        if not hasattr(self, 'query_path'):
            QMessageBox.warning(self, "提示", "请先选择查询图片")
            return

        # 搜索时先显示“正在检索…”
        for label in self.result_labels:
            label.setText("检索中...")
            label.setPixmap(QPixmap())

        # 创建异步线程
        self.thread = SearchThread(
            self.model, self.transform,
            self.gallery_paths, self.query_path,
            self.device
        )
        self.thread.finished.connect(self.show_results)
        self.thread.start()

    # ==========================
    # 显示结果
    # ==========================
    def show_results(self, topk_paths):
        for i, label in enumerate(self.result_labels):
            pixmap = QPixmap(topk_paths[i]).scaled(
                label.size(), Qt.KeepAspectRatio
            )
            label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ReIDApp()
    win.show()
    sys.exit(app.exec_())