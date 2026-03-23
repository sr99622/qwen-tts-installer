import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class BatchSelectionDialog(QDialog):
    def __init__(self, batch_dirs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Batch Folder")
        self.resize(700, 450)
        self.selected_batch_path = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Available batch folders"))

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget, 1)

        for batch_dir in batch_dirs:
            item = QListWidgetItem(os.path.basename(batch_dir))
            item.setData(Qt.ItemDataRole.UserRole, batch_dir)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsEditable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            self.list_widget.addItem(item)

        self.list_widget.itemDoubleClicked.connect(self.accept_selection)
        self.list_widget.itemChanged.connect(self.on_item_changed)

        rename_layout = QHBoxLayout()
        self.rename_button = QPushButton("Rename Selected")
        self.rename_button.clicked.connect(self.rename_selected)
        rename_layout.addWidget(self.rename_button)
        rename_layout.addStretch()
        layout.addLayout(rename_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def rename_selected(self):
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "No Selection", "Please select a batch folder.")
            return
        self.list_widget.editItem(item)

    def accept_selection(self):
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "No Selection", "Please select a batch folder.")
            return

        self.selected_batch_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def on_item_changed(self, item: QListWidgetItem):
        import json

        old_path = item.data(Qt.ItemDataRole.UserRole)
        if not old_path:
            return

        parent_dir = os.path.dirname(old_path)
        old_name = os.path.basename(old_path)
        new_name = item.text().strip()

        if not new_name:
            QMessageBox.warning(self, "Invalid Name", "Folder name cannot be empty.")
            item.setText(old_name)
            return

        if new_name == old_name:
            return

        if "/" in new_name or "\\" in new_name:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Folder name cannot contain path separators.",
            )
            item.setText(old_name)
            return

        new_path = os.path.join(parent_dir, new_name)

        if os.path.exists(new_path):
            QMessageBox.warning(
                self,
                "Name Already Exists",
                f"A folder with this name already exists:\n\n{new_name}",
            )
            item.setText(old_name)
            return

        try:
            os.rename(old_path, new_path)

            manifest_path = os.path.join(new_path, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                manifest["batch_name"] = new_name
                manifest["batch_dir"] = new_path

                if isinstance(manifest.get("files"), list):
                    for file_item in manifest["files"]:
                        old_file_path = file_item.get("path", "")
                        if old_file_path:
                            file_item["path"] = os.path.join(
                                new_path,
                                os.path.basename(old_file_path),
                            )

                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)

            item.setData(Qt.ItemDataRole.UserRole, new_path)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Rename Failed",
                f"Could not rename folder:\n\n{e}",
            )
            item.setText(old_name)
