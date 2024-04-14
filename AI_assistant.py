import tkinter as tk
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import io
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2

import pickle

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light")
customtkinter.set_default_color_theme("green")

class Net(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(59536, 64)
        self.fc2 = nn.Linear(64, 2)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc3 = nn.Linear(1, 3)  # 1-dimensional output from QNN

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)
        return x

def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.image_label = customtkinter.CTkLabel(self)
        self.image_label.grid(row=0, column=1) 
        self.image_label.configure(text="")
        self.second_image_label = customtkinter.CTkLabel(self) 
        self.second_image_label.grid(row=0, column=2)
        self.second_image_label.configure(text="") 

        image_width = 300
        
        self.result_label = customtkinter.CTkLabel(self, text="", font=customtkinter.CTkFont(size=22, weight="bold"),
                                                   text_color=("black", "black"), width=image_width)
        self.result_label.grid(row=2, column=1) 

        # configure window
        self.title("Breast cancer ultrasound image analyser")
        self.geometry(f"{580}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure((1, 2), weight=1)
        self.grid_columnconfigure((3, 4), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="QLFuture",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.browse_image)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.analyse_image)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        self.sidebar_button_1.configure(text="Upload image")
        self.sidebar_button_2.configure(text="Analyse")

        self.model_weights_path = 'model/model.pth'
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        qnn = create_qnn()
        self.model = Net(qnn).to(self.device)

        weights = torch.load(self.model_weights_path)
        self.model.load_state_dict(weights)

    def browse_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            photo = customtkinter.CTkImage(Image.open(self.file_path), size=(300, 300))
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.result_label.configure(text="")
            self.clear_second_image()

    def analyse_image(self):
        if hasattr(self, 'image_label') and hasattr(self.image_label, 'image'):
            # file_path = getattr(self.image_label, 'image', None)
            if self.file_path:
                img = cv2.resize(np.array(Image.open(self.file_path)), (256,256))/255
                img = torch.from_numpy(img).to(self.device).float()
                img = img.permute(2, 0, 1)
                img = img.unsqueeze(0)
                
                self.model.eval() 
                with torch.no_grad():
                    output = self.model(img)

                predicted_class, prob = torch.argmax(output, dim=1).item(), round(torch.max(output, dim=1).values.item(), 2)
                
                if predicted_class == 0:
                    self.result_label.configure(text=f"Classification: Benign\nCertainty: {prob}", text_color=("yellow", "yellow"))
                elif predicted_class == 1:
                    self.result_label.configure(text=f"Classification: Normal\nCertainty: {prob}", text_color=("green", "green"))
                else:
                    self.result_label.configure(text=f"Classification: Malignant\nCertainty: {prob}", text_color=("red", "red"))

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def clear_second_image(self):
        self.second_image_label.configure(image=None)
        self.second_image_label.image = None


if __name__ == "__main__":
    app = App()
    app.mainloop()
