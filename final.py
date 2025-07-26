# === FINAL AND COMPLETE CODE ===

# --- All Necessary Imports ---
import numpy as np
import math
import cv2
import os
import sys
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import tkinter as tk
from PIL import Image, ImageTk
import customtkinter

# --- Initial Setup (from your original code) ---
ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
offset = 29
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

# --- Main Application Class ---
class Application:
    def __init__(self):
        # --- CORE APP SETUP (Same as your original code) ---
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('cnn8grps_rad1_model.h5')
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice", voices[0].id)
        
        self.ct = {'blank': 0}
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10
        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")

        self.str = " "
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        self.ccc = 0

        # --- MODERN GUI SETUP ---
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")

        self.root = customtkinter.CTk()
        self.root.title("AI Sign Language Interpreter")
        self.root.geometry("1200x720")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        # Configure the grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # --- HEADER ---
        self.header_frame = customtkinter.CTkFrame(self.root, corner_radius=0)
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        self.T = customtkinter.CTkLabel(self.header_frame, text="AI Sign Language Interpreter", font=customtkinter.CTkFont(size=28, weight="bold"))
        self.T.grid(row=0, column=0, padx=20, pady=10)

        # --- LEFT PANEL (VIDEO FEED) ---
        self.left_panel = customtkinter.CTkFrame(self.root, width=640, height=480)
        self.left_panel.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="nsew")
        self.left_panel.grid_propagate(False)
        self.left_panel.grid_rowconfigure(0, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        self.panel = customtkinter.CTkLabel(self.left_panel, text="")
        self.panel.grid(row=0, column=0, sticky="nsew")

        # --- RIGHT PANEL (SKELETON & OUTPUT) ---
        self.right_panel = customtkinter.CTkFrame(self.root)
        self.right_panel.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1)

        self.panel2 = customtkinter.CTkLabel(self.right_panel, text="")
        self.panel2.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)

        self.char_frame = customtkinter.CTkFrame(self.right_panel, fg_color="transparent")
        self.char_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky="ew")
        self.char_frame.grid_columnconfigure(1, weight=1)
        self.T1 = customtkinter.CTkLabel(self.char_frame, text="Character:", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.T1.grid(row=0, column=0, padx=(20, 10))
        self.panel3 = customtkinter.CTkLabel(self.char_frame, text="", font=customtkinter.CTkFont(size=20))
        self.panel3.grid(row=0, column=1, sticky="w")
        
        self.sentence_frame = customtkinter.CTkFrame(self.right_panel, fg_color="transparent")
        self.sentence_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20), sticky="ew")
        self.sentence_frame.grid_columnconfigure(1, weight=1)
        self.T3 = customtkinter.CTkLabel(self.sentence_frame, text="Sentence:", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.T3.grid(row=0, column=0, padx=(20, 10))
        self.panel5 = customtkinter.CTkLabel(self.sentence_frame, text="", font=customtkinter.CTkFont(size=20), wraplength=400, justify="left")
        self.panel5.grid(row=0, column=1, sticky="w")
        
        # --- FOOTER (SUGGESTIONS & ACTIONS) ---
        self.footer_frame = customtkinter.CTkFrame(self.root)
        self.footer_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew")
        self.footer_frame.grid_columnconfigure(0, weight=1)

        self.suggestions_label = customtkinter.CTkLabel(self.footer_frame, text="Suggestions:", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.suggestions_label.grid(row=0, column=0, padx=20, pady=(10, 5), sticky="w")

        self.suggestions_bar = customtkinter.CTkFrame(self.footer_frame, fg_color="transparent")
        self.suggestions_bar.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        
        self.b1 = customtkinter.CTkButton(self.suggestions_bar, text="", command=self.action1, width=120)
        self.b1.pack(side="left", padx=5)
        self.b2 = customtkinter.CTkButton(self.suggestions_bar, text="", command=self.action2, width=120)
        self.b2.pack(side="left", padx=5)
        self.b3 = customtkinter.CTkButton(self.suggestions_bar, text="", command=self.action3, width=120)
        self.b3.pack(side="left", padx=5)
        self.b4 = customtkinter.CTkButton(self.suggestions_bar, text="", command=self.action4, width=120)
        self.b4.pack(side="left", padx=5)
        
        self.action_buttons_frame = customtkinter.CTkFrame(self.footer_frame, fg_color="transparent")
        self.action_buttons_frame.grid(row=1, column=1, padx=20, pady=(0, 10), sticky="e")
        
        self.clear = customtkinter.CTkButton(self.action_buttons_frame, text="Clear", command=self.clear_fun, width=100)
        self.clear.pack(side="right", padx=5)
        self.speak = customtkinter.CTkButton(self.action_buttons_frame, text="Speak", command=self.speak_fun, width=100)
        self.speak.pack(side="right", padx=5)

        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                self.root.after(1, self.video_loop)
                return

            cv2image = cv2.flip(frame, 1)
            
            img_rgb = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(img_rgb)
            imgtk = customtkinter.CTkImage(light_image=self.current_image, dark_image=self.current_image, size=(640, 480))
            self.panel.configure(image=imgtk)
            
            hands, _ = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy = np.array(cv2image)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                y1 = max(0, y - offset)
                y2 = min(cv2image_copy.shape[0], y + h + offset)
                x1 = max(0, x - offset)
                x2 = min(cv2image_copy.shape[1], x + w + offset)
                image = cv2image_copy[y1:y2, x1:x2]

                # Using a 3-channel white image for compatibility with color drawing
                white = np.ones((400, 400, 3), np.uint8) * 255
                
                if image.size > 0:
                    handz, _ = hd2.findHands(image, draw=False, flipType=True)
                    if handz:
                        hand = handz[0]
                        self.pts = hand['lmList']
                        
                        os_x = ((400 - w) // 2) - 15
                        os_y = ((400 - h) // 2) - 15
                        
                        # Drawing skeleton logic from your original file
                        for t in range(0, 4, 1): cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(5, 8, 1): cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(9, 12, 1): cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(13, 16, 1): cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(17, 20, 1): cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y), (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y), (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                        for i in range(21): cv2.circle(white, (self.pts[i][0] + os_x, self.pts[i][1] + os_y), 2, (0, 0, 255), 1)

                        res = white
                        self.predict(res)
                        
                        self.current_image2 = Image.fromarray(res)
                        imgtk_white = customtkinter.CTkImage(light_image=self.current_image2, dark_image=self.current_image2, size=(400, 400))
                        self.panel2.configure(image=imgtk_white)
            else:
                self.panel2.configure(image=None)
                self.current_symbol = ""
            
            self.panel3.configure(text=self.current_symbol)
            self.b1.configure(text=self.word1)
            self.b2.configure(text=self.word2)
            self.b3.configure(text=self.word3)
            self.b4.configure(text=self.word4)
            self.panel5.configure(text=self.str)

        except Exception as e:
            print(f"Error in video loop: {e}")
            traceback.print_exc()
        finally:
            self.root.after(10, self.video_loop)

    def predict(self, test_image):
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 0
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 4
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                ch1 = 5
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6
        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1
        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 1
        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1
        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 1
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 7
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + 13 < self.pts[8][0] and self.pts[0][0] + 13 < self.pts[12][0] and self.pts[0][0] + 13 < self.pts[16][0] and self.pts[0][0] + 13 < self.pts[20][0]) and not (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1
        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]: ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]: ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]: ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]: ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]: ch1 = 'N'
        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42: ch1 = 'C'
            else: ch1 = 'O'
        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72: ch1 = 'G'
            else: ch1 = 'H'
        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42: ch1 = 'Y'
            else: ch1 = 'J'
        if ch1 == 4: ch1 = 'L'
        if ch1 == 6: ch1 = 'X'
        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]: ch1 = 'Z'
                else: ch1 = 'Q'
            else: ch1 = 'P'
        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]: ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]): ch1 = 'V'
            if (self.pts[8][0] > self.pts[12][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): ch1 = 'R'
        if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): ch1 = " "
        if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): ch1 = "next"
        if ch1 == 'Next' or ch1 == 'B' or ch1 == 'C' or ch1 == 'H' or ch1 == 'F' or ch1 == 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]): ch1 = 'Backspace'
        if ch1 == "next" and self.prev_char != "next":
            if self.ten_prev_char[(self.count - 2) % 10] != "next":
                if self.ten_prev_char[(self.count - 2) % 10] == "Backspace": self.str = self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace": self.str = self.str + self.ten_prev_char[(self.count - 2) % 10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace": self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]
        if ch1 == "  " and self.prev_char != "  ": self.str = self.str + "  "
        self.prev_char = ch1
        self.current_symbol = ch1
        self.count += 1
        self.ten_prev_char[self.count % 10] = ch1
        if len(self.str.strip()) != 0:
            st = self.str.rfind(" ")
            ed = len(self.str)
            word = self.str[st + 1:ed]
            self.word = word
            if len(word.strip()) != 0:
                ddd.check(word)
                lenn = len(ddd.suggest(word))
                if lenn >= 4: self.word4 = ddd.suggest(word)[3]
                if lenn >= 3: self.word3 = ddd.suggest(word)[2]
                if lenn >= 2: self.word2 = ddd.suggest(word)[1]
                if lenn >= 1: self.word1 = ddd.suggest(word)[0]
            else:
                self.word1 = " "
                self.word2 = " "
                self.word3 = " "
                self.word4 = " "

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()

    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word2.upper()

    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()

    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()

    def speak_fun(self):
        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str = " "
        self.word = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

# --- Start the Application ---
if __name__ == "__main__":
    print("Starting Application...")
    app = Application()
    app.root.mainloop()
