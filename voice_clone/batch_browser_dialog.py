import json
import os
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QVBoxLayout,
)


class BatchBrowserDialog(QDialog):
    def __init__(self, output_root: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Batch Folder")
        self.resize(700, 420)

        self.output_root = output_root
        self.selected_batch_dir: Optional[str] = None

        layout = QHBoxLayout(self)

        self.list_widget = QListWidget()
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)

        layout.addWidget(self.list_widget, 1)
        layout.addWidget(self.preview, 2)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)

        outer = QVBoxLayout()
        outer.addLayout(layout)
        outer.addWidget(button_box)

        wrapper = QVBoxLayout()
        wrapper.addLayout(outer)
        self.setLayout(wrapper)

        self.list_widget.currentItemChanged.connect(self.update_preview)
        self.populate()

    def populate(self):
        self.list_widget.clear()
        if not os.path.isdir(self.output_root):
            return

        dirs = []
        for name in os.listdir(self.output_root):
            full = os.path.join(self.output_root, name)
            if os.path.isdir(full):
                dirs.append((os.path.getmtime(full), name, full))

        dirs.sort(reverse=True)

        for _, name, full in dirs:
            item = QListWidgetItem(name)
            item.setData(256, full)
            self.list_widget.addItem(item)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def update_preview(self, current: QListWidgetItem, previous: QListWidgetItem):
        _ = previous
        if not current:
            self.preview.clear()
            return

        batch_dir = current.data(256)
        metadata_path = os.path.join(batch_dir, "batch_metadata.json")
        script_path = os.path.join(batch_dir, "script.txt")

        parts = [f"Folder: {os.path.basename(batch_dir)}"]

        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                parts.append(f"Model: {data.get('model_name', '')}")
                parts.append(f"Language: {data.get('language', '')}")
                parts.append(f"Batch size: {data.get('batch_size', '')}")
                parts.append("")
            except Exception:
                parts.append("Metadata could not be read.")
                parts.append("")

        if os.path.isfile(script_path):
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    script = f.read().strip()
                parts.append("Script:")
                parts.append(script[:2000])
            except Exception:
                parts.append("Script could not be read.")

        self.preview.setPlainText("\n".join(parts))

    def accept_selection(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        self.selected_batch_dir = item.data(256)
        self.accept()
