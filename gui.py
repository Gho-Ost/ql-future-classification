import tkinter as tk
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
from classifier import classify_image  # Import funkcji z zewnętrznego pliku

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light")
customtkinter.set_default_color_theme("green")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.image_label = customtkinter.CTkLabel(self)
        self.image_label.grid(row=0, column=1)  # Zmodyfikuj odpowiednio, jeśli to potrzebne
        self.image_label.configure(text="")  # Początkowo pusty tekst
        self.second_image_label = customtkinter.CTkLabel(self)  # Dodajemy drugi label dla drugiego obrazka
        self.second_image_label.grid(row=0, column=2)  # Zmieniony kolumnę na 2
        self.second_image_label.configure(text="")  # Początkowo pusty tekst

        # Oblicz szerokość obrazka
        image_width = 300  # Przykładowa szerokość, należy dostosować do rzeczywistych wymiarów obrazka

        # Ustaw szerokość etykiety result_label na szerokość obrazka
        self.result_label = customtkinter.CTkLabel(self, text="", font=customtkinter.CTkFont(size=22, weight="bold"),
                                                   text_color=("black", "black"), width=image_width)
        self.result_label.grid(row=1, column=2)  # Przeniesiona do drugiej kolumny

        # configure window
        self.title("Breast cancer ultrasound image analyser")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure((1, 2), weight=1)  # Zmiana kolumn dla obrazków i wyniku
        self.grid_columnconfigure((3, 4), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)  # Zmiana wierszy dla obrazków i wyniku

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

    def browse_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            photo = customtkinter.CTkImage(Image.open(file_path), size=(300, 300))
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.result_label.configure(text="")  # Wyczyść poprzedni wynik
            # Nie wywołujemy tutaj funkcji classify_image, ponieważ chcemy, aby została wywołana przy analizie
            self.clear_second_image()  # Wywołanie funkcji czyszczącej drugi obraz

    def analyse_image(self):
        if hasattr(self, 'image_label') and hasattr(self.image_label, 'image'):
            file_path = getattr(self.image_label, 'image', None)
            if file_path:
                result = classify_image(file_path)  # Uruchomienie klasyfikatora

                # Wczytanie obrazu 'pobrane.jpg' i wyświetlenie go
                second_photo = customtkinter.CTkImage(Image.open('pobrane.jpg'), size=(300, 300))
                self.second_image_label.configure(image=second_photo)
                self.second_image_label.image = second_photo

                if result == 'rak:(':
                    self.result_label.configure(text="Klasyfikacja: " + result, text_color=("red", "red"))
                else:
                    self.result_label.configure(text="Klasyfikacja: " + result, text_color=("green", "green"))

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def clear_second_image(self):
        # Funkcja czyszcząca drugi obraz
        self.second_image_label.configure(image=None)
        self.second_image_label.image = None


if __name__ == "__main__":
    app = App()
    app.mainloop()
