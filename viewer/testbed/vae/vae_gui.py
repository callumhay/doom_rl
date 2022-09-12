from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout
from PyQt6.QtCore import QProcess
import PyQt6.QtCore as QtCore
import sys
import typing

class MainWindow(QMainWindow):
  def __init__(self, parent: typing.Optional[QWidget] = None, flags: QtCore.Qt.WindowType = QtCore.Qt.WindowType.Window) -> None:
    super().__init__(parent, flags)
    self.vae_proc = None
    self.btn = QPushButton("Start")
    self.btn.pressed.connect(self.start_vae_trainer_proc)
    
    layout = QVBoxLayout()
    layout.addWidget(self.btn)
    
    central_widget = QWidget()
    central_widget.setLayout(layout)
    
    self.setCentralWidget(central_widget)
    

  def start_vae_trainer_proc(self):
    if self.vae_proc != None: return
    self.vae_proc = QProcess()
    self.vae_proc.start("python", ["./vae_main.py"]) # TODO: Map options
    self.btn.setEnabled(False)
    
  def clean_up(self):
    if self.vae_proc != None: self.vae_proc.kill()

if __name__ =="__main__":
  app = QApplication(sys.argv)
  w = MainWindow()
  w.show()
  
  app.exec()
  w.clean_up()
  