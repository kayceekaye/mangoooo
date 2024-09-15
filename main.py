import tkinter as tk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from pathlib import Path

class Application(tk.Frame):
    def __init__(self, master=None, camera_index=0):
        super().__init__(master)
        self.master = master
        self.camera_index = camera_index
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.camera_label = tk.Label(self)
        self.camera_label.pack()

        self.snapshot_button = tk.Button(self)
        self.snapshot_button["text"] = "Snapshot!"
        self.snapshot_button["command"] = self.take_snapshot
        self.snapshot_button.pack(side="bottom")

        self.snapshot_frame = tk.Frame(self)
        self.snapshot_frame.pack(side="top", fill="both", expand=True)

        self.snapshot_label = tk.Label(self.snapshot_frame)
        self.snapshot_label.pack()
        
        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

        self.video_stream = cv2.VideoCapture(self.camera_index)
        self.update_camera()

    def update_camera(self):
        ret, frame = self.video_stream.read()
        if ret:
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_im)
            img_tk = ImageTk.PhotoImage(image=img)
            self.camera_label.config(image=img_tk)
            self.camera_label.image = img_tk
        self.after(1, self.update_camera)

    def take_snapshot(self):
        ret, frame = self.video_stream.read()
        if ret:
            cv2.imwrite("snapshot.jpg", frame)
            print("Snapshot saved!")
            model = YOLO('runs/detect/train3/weights/last.pt')
            results = model(source='snapshot.jpg', conf=0.4, save=True)
            
            # Create a Path object for the directory
            path = Path('runs/detect')
            # Count only directories
            folder_count = sum(1 for entry in path.iterdir() if entry.is_dir())
            total_predict_count = folder_count - 1
            print(total_predict_count)
            
            # Load the snapshot image
            img = Image.open(f"runs/detect/predict{total_predict_count}/snapshot.jpg")
            img_tk = ImageTk.PhotoImage(image=img)

            # Display the snapshot image
            self.snapshot_label.config(image=img_tk)
            self.snapshot_label.image = img_tk

root = tk.Tk()
app = Application(master=root, camera_index=1)  # Use camera index 1 for the second camera
app.mainloop()
