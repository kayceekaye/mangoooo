import tkinter as tk
import cv2
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.camera_label = tk.Label(self)
        self.camera_label.pack()

        self.snapshot_button = tk.Button(self)
        self.snapshot_button["text"] = "Snapshot!"
        self.snapshot_button["command"] = self.take_snapshot
        self.snapshot_button.pack(side="bottom")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

        self.video_stream = cv2.VideoCapture(0)
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

root = tk.Tk()
app = Application(master=root)
app.mainloop()