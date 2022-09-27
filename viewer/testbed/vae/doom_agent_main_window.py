import sys
import typing

import torch
import torchvision.transforms.functional as F
#import torchshow as ts

from PIL.ImageQt import ImageQt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSpacerItem, QSizePolicy, QCheckBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import pyqtSlot, pyqtSignal, Qt
import PyQt6.QtWidgets as QtWidgets
import PyQt6.QtCore as QtCore
import PyQt6.QtGui as QtGui

from doom_agent_thread import DoomAgentThread, DEVICE
from doom_env import DoomEnv, PREPROCESS_RES_H_W

class DoomAgentMainWindow(QMainWindow):
  # Signals
  lr_changed_signal = pyqtSignal(float, float, name="lr_changed")
  batches_per_action_changed_signal = pyqtSignal(int, name="batches_per_action_changed")

  def __init__(self, parent: typing.Optional[QWidget] = None, flags: QtCore.Qt.WindowType = QtCore.Qt.WindowType.Window) -> None:
    super().__init__(parent, flags)
    
    self.setFixedSize(800, 700)
    
    self.doom_agent = DoomAgentThread()
    self.doom_agent.screenbuf_signal.connect(self.update_screenbuf_slot)
    self.doom_agent.screenbuf_available_signal.connect(self.on_doom_agent_started)
    self.doom_agent.vae_saved_signal.connect(self.rebuild_doom_vae)
    self.doom_agent.finished.connect(self.on_doom_agent_finished)
    self.lr_changed_signal.connect(self.doom_agent.set_lr_min_max)
    self.batches_per_action_changed_signal.connect(self.doom_agent.set_batches_per_action)

    # VAE network for GUI visualization (training is done by the DoomAgentThread)
    self.doom_vae = self.doom_agent.build_vae_network()
    self.doom_vae.eval()
    
    self.input_screenbuf_tensor  = None

    self.request_screenbuf_btn = QPushButton("Capture Screen")
    self.request_screenbuf_btn.pressed.connect(self.doom_agent.request_screenbuf)
    self.request_screenbuf_btn.setEnabled(False)
    self.screenbuf_fps_spinbox = QtWidgets.QSpinBox()
    self.screenbuf_fps_spinbox.setMinimum(0)
    self.screenbuf_fps_spinbox.setMaximum(35)
    self.screenbuf_fps_spinbox.setValue(self.doom_agent.screenbuf_signal_fps)
    self.screenbuf_fps_spinbox.valueChanged.connect(self.doom_agent.set_fps)
    self.screenbuf_fps_spinbox.valueChanged.connect(lambda x: self.request_screenbuf_btn.setEnabled(False if x==0 else True))
    
    self.start_agent_btn = QPushButton("Start Agent")
    self.start_agent_btn.pressed.connect(self.start_stop_agent_slot)
    
    img_h, img_w = PREPROCESS_RES_H_W
    def build_img_label(w, h):
      img_label = QLabel()
      img_label.setBaseSize(w, h)
      img_label.setMinimumSize(w, h)
      return img_label
    
    self.input_img_label  = build_img_label(2*img_w, 2*img_h)
    self.output_img_label = build_img_label(2*img_w, 2*img_h)
    
    label_grid_widget = QWidget()
    label_grid_layout = QtWidgets.QGridLayout()
    self.output_img_labels = []
    output_img_grid_w = 3
    output_img_grid_h = 3
    output_img_h = int(2*img_h/output_img_grid_h)
    output_img_w = int(2*img_w/output_img_grid_w)
    for x in range(output_img_grid_w):
      for y in range(output_img_grid_h):
        img_label = build_img_label(output_img_w, output_img_h)
        self.output_img_labels.append(img_label)
        label_grid_layout.addWidget(img_label, x, y)
    
    label_grid_widget.setLayout(label_grid_layout)
    
    output_img_widget = QWidget()
    output_top_layout = QVBoxLayout()
    output_img_widget.setLayout(output_top_layout)
    output_top_layout.addWidget(self.output_img_label, alignment=Qt.AlignmentFlag.AlignCenter)
    output_top_layout.addWidget(label_grid_widget, alignment=Qt.AlignmentFlag.AlignCenter)
    output_top_layout.addSpacerItem(QSpacerItem(20,40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
    
    output_controls_widget = QWidget()
    output_controls_layout = QHBoxLayout()
    output_controls_widget.setLayout(output_controls_layout)
    output_controls_layout.addSpacerItem(QSpacerItem(40,20,QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
    output_controls_layout.addWidget(QLabel("fps:"))
    output_controls_layout.addWidget(self.screenbuf_fps_spinbox)
    output_controls_layout.addWidget(self.request_screenbuf_btn)
    output_top_layout.addWidget(output_controls_widget)
    output_controls_layout.addSpacerItem(QSpacerItem(40,20,QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
    
    img_display_widget = QWidget()
    img_layout = QHBoxLayout()
    img_display_widget.setLayout(img_layout)
    img_layout.addWidget(self.input_img_label, alignment=Qt.AlignmentFlag.AlignCenter)
    img_layout.addSpacerItem(QSpacerItem(40,20,QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
    img_layout.addWidget(output_img_widget, alignment=Qt.AlignmentFlag.AlignCenter)
    
    self.lr_min_widget = QtWidgets.QDoubleSpinBox()
    self.lr_min_widget.setDecimals(9)
    self.lr_min_widget.setStepType(QtWidgets.QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
    self.lr_min_widget.setMinimum(0.000000001)
    self.lr_min_widget.setMaximum(0.1)
    self.lr_min_widget.setValue(self.doom_agent.lr_min)
    self.lr_min_widget.editingFinished.connect(self.update_lr)
    self.lr_min_widget.setEnabled(True)
    
    self.lr_max_widget = QtWidgets.QDoubleSpinBox()
    self.lr_max_widget.setDecimals(9)
    self.lr_max_widget.setStepType(QtWidgets.QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
    self.lr_max_widget.setMinimum(0.000000001)
    self.lr_max_widget.setMaximum(0.1)
    self.lr_max_widget.setValue(self.doom_agent.lr_max)
    self.lr_max_widget.editingFinished.connect(self.update_lr)
    self.lr_max_widget.setEnabled(True)

    self.batches_per_action_widget = QtWidgets.QSpinBox()
    self.batches_per_action_widget.setMinimum(1)
    self.batches_per_action_widget.setMaximum(1000)
    self.batches_per_action_widget.setSingleStep(1)
    self.batches_per_action_widget.setValue(self.doom_agent.batches_per_action)
    self.batches_per_action_widget.editingFinished.connect(self.update_batches_per_action)
    self.batches_per_action_widget.setEnabled(True)
    
    self.toggle_play_chkbox = QCheckBox("Enable Play")
    self.toggle_play_chkbox.setChecked(True)
    self.toggle_play_chkbox.toggled.connect(self.doom_agent.toggle_play)

    btn_widget = QWidget()
    btn_layout = QHBoxLayout()
    btn_widget.setLayout(btn_layout)
    btn_layout.addWidget(self.start_agent_btn)
    btn_layout.addWidget(QLabel("LR Min:"))
    btn_layout.addWidget(self.lr_min_widget)
    btn_layout.addWidget(QLabel("LR Max:"))
    btn_layout.addWidget(self.lr_max_widget)
    btn_layout.addWidget(QLabel("Batches per Action:"))
    btn_layout.addWidget(self.batches_per_action_widget)
    btn_layout.addWidget(self.toggle_play_chkbox)
    btn_layout.addSpacerItem(QSpacerItem(40,20,QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
    
    top_level_layout = QVBoxLayout()
    top_level_layout.addWidget(img_display_widget)
    top_level_layout.addSpacerItem(QSpacerItem(20,40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
    top_level_layout.addWidget(btn_widget)
    
    central_widget = QWidget()
    central_widget.setLayout(top_level_layout)
    self.setCentralWidget(central_widget)
    
    
  def _update_label_image_from_screenbuf(label, tensor):
    tensor = tensor.squeeze(0) if tensor.ndim == 4 else tensor
    orig_tensor = DoomEnv.deprocess_screenbuffer(tensor).to(torch.device('cpu'))
    qpixmap = QPixmap.fromImage(ImageQt(F.to_pil_image(orig_tensor)))
    label.setPixmap(qpixmap.scaled(label.width(),label.height(), Qt.AspectRatioMode.KeepAspectRatio))
    label.repaint()
    
  # Qt Slots *******************
  @pyqtSlot()
  def rebuild_doom_vae(self):
    self.doom_vae = self.doom_agent.build_vae_network()
    self.doom_vae.eval()
  @pyqtSlot(torch.Tensor)
  def update_screenbuf_slot(self, screenbuf_tensor):
    if len(screenbuf_tensor.shape) != 3: return
    self.input_screenbuf_tensor = screenbuf_tensor.to(DEVICE).unsqueeze_(0)
    DoomAgentMainWindow._update_label_image_from_screenbuf(self.input_img_label, screenbuf_tensor)
    # Run the input screen buffer through the Doom VAE network and show what the encoded to decoded output looks like 
    
    #output_screenbuf_tensors = self.doom_vae.generate(
    output_screenbuf_tensors, _ = self.doom_vae( 
      self.input_screenbuf_tensor.expand(len(self.output_img_labels)+1,*screenbuf_tensor.shape)
    )
    for i in range(len(self.output_img_labels)):
      DoomAgentMainWindow._update_label_image_from_screenbuf(self.output_img_labels[i], output_screenbuf_tensors[i])
    #DoomAgentMainWindow._update_label_image_from_screenbuf(self.output_img_label, output_screenbuf_tensors[0])
    mean_screenbuf_tensor, _ = self.doom_vae(self.input_screenbuf_tensor, False)
    DoomAgentMainWindow._update_label_image_from_screenbuf(self.output_img_label, mean_screenbuf_tensor)
    
  @pyqtSlot()
  def update_lr(self):
    b1 = QtCore.QSignalBlocker(self.lr_max_widget)
    b2 = QtCore.QSignalBlocker(self.lr_min_widget)
    
    sender = self.sender()
    if sender == self.lr_max_widget:
      self.lr_min_widget.setValue(min(self.lr_max_widget.value(), self.lr_min_widget.value()))
      self.lr_max_widget.setValue(max(self.lr_max_widget.value(), self.lr_min_widget.value()))
    elif sender == self.lr_min_widget:
      self.lr_max_widget.setValue(max(self.lr_max_widget.value(), self.lr_min_widget.value()))  
      self.lr_min_widget.setValue(min(self.lr_max_widget.value(), self.lr_min_widget.value()))
    
    self.lr_changed_signal.emit(self.lr_min_widget.value(), self.lr_max_widget.value())
  @pyqtSlot()
  def update_batches_per_action(self):
    b = QtCore.QSignalBlocker(self.batches_per_action_widget)
    self.batches_per_action_changed_signal.emit(self.batches_per_action_widget.value())
    
  @pyqtSlot()
  def start_stop_agent_slot(self):
    self.start_agent_btn.setEnabled(False)
    if self.doom_agent.isRunning():
      self.doom_agent.quit()
      QtCore.QMetaObject.invokeMethod(self.doom_agent, "stop")
    else:
      self.doom_agent.start()
  @pyqtSlot()
  def on_doom_agent_started(self):
    self.start_agent_btn.setText("Kill Agent")
    self.start_agent_btn.setEnabled(True)
    self.request_screenbuf_btn.setEnabled(True)
    self.lr_min_widget.setEnabled(True)
    self.lr_max_widget.setEnabled(True)
    self.batches_per_action_widget.setEnabled(True)
  @pyqtSlot()
  def on_doom_agent_finished(self):
    self.start_agent_btn.setText("Start Agent")
    self.start_agent_btn.setEnabled(True)
    self.request_screenbuf_btn.setEnabled(False)
    self.lr_min_widget.setEnabled(True)
    self.lr_max_widget.setEnabled(True)
    self.batches_per_action_widget.setEnabled(True)
  # End Qt Slots *******************************
    
  
  def terminate(self):
    QtCore.QMetaObject.invokeMethod(self.doom_agent, "stop")
    self.doom_agent.wait()

if __name__ =="__main__":
  app = QApplication(sys.argv)
  
  w = DoomAgentMainWindow()
  w.show()
  
  app.exec()
  w.terminate()

  