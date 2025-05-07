import tkinter as tk
from tkinter import PhotoImage
from tkinter import *
from PIL import Image, ImageTk
import os
import shutil
import threading
import tkinter.messagebox as tkm
from random import randint
import tensorflow as tf
from multiprocessing import Process
from wav_reader_f import get_fft_spectrum
from keras.models import load_model
import keras.losses
import keras.backend as K
import numpy as np
from random import randint
import constants_f as c
import os
from threading import *
import time
import threading
from keras import backend as K
# import tensorflow.keras.backend as K
import shutil
from recorder_f import Recorder
import Funzioni_per_la_registrazione_audio as f_audio
from pydub import AudioSegment
from joblib import dump, load
import librosa

from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from keras.models import model_from_json

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, Page_Voce, Page_Volto, Page_Voce_e_Volto):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame


            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.show_frame("StartPage")

    def on_exit(self):
        """When you click to exit, this function is called"""
        if tkm.askyesno("Exit", "Would you really exit?"):
            print("SONO ENTRATO IN CHIUSURA INTERFACCIA.")
            flag[0] = False

            # Elimino eventuali thread ancora in vita
            main_thread = threading.currentThread()
            for t in threading.enumerate():
                if t is main_thread:
                    continue
                print("Nome thread:", t.getName())
                t.join()
                print("Eliminato.")
            file_report.write("\nEND PROGRAM\n")
            file_report.write(time.strftime("Time: %H:%M:%S\n"))
            file_report.close()

            #Con il codice di sotto elimino eventuali file aggiunti dai thread che non sono stati cancellati correttamente.
            for file in os.listdir():
                if (file[:10] == 'audio_file'):
                    os.unlink(file)
            self.destroy()

    def show_frame(self, page_name):

        frame = self.frames[page_name]
        frame.tkraise()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        flag[0] = True
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.winfo_toplevel().title("EMOFEATURES")
        self.configure(background="midnight blue")

        intro = tk.Label(self, height=5, width=50, text="EMOFEATURES\n\nEmotions recognition through face and voice analysis", bg="green yellow", font = ('black', 15, 'bold'))
        b_ver = tk.Button(self, height=2, width=40, text="Emotions Recognition Through Voice", highlightbackground="gray", bg="white",
                          activebackground="gray", command=lambda: controller.show_frame("Page_Voce"))

        b_id = tk.Button(self, height=2, width=40, text="Emotions Recognition Through Face", bg="white", highlightbackground="gray", activebackground="blue",
                         command=lambda: controller.show_frame("Page_Volto"))

        b_registrazione = tk.Button(self, height=2, width=40, text="Emotions Recognition Through Face and Voice", bg="white", highlightbackground="gray", activebackground="gray" ,command=lambda: controller.show_frame("Page_Voce_e_Volto"))




        intro.grid(row=0, column=0, sticky="n", pady=20, padx=45)
        b_ver.grid(row=1, column=0, sticky="n", pady=40)
        b_id.grid(row=1, column=0, sticky="n", pady=120)
        b_registrazione.grid(row=1, column=0, sticky="n", pady=200)


        #button.pack()

    #def Aggiorna_Utenti_Registrati(self):
     #   listbox = Listbox(self, )

############################################################################
#                   INIZIO RICONOSCIMENTO SOLO DALLA VOCE                  #
############################################################################
class Page_Voce(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(background="midnight blue")

        label_benvenuto = Label(self, width=50, font=('black', 13, 'bold'), bg="green yellow", text="Welcome to Emotions Recognition through Voice", pady=20, padx=30)

        indietro = tk.Button(self, text="Back", bg="white", highlightbackground="white", activebackground="white", width=10, command=lambda: Pulisci_finestra_voce(self, avvia_ric, indietro))

        blocca_registrazione = Button(self, width=10, text="Stop", bg="white", highlightbackground ="gray", activebackground="gray", command=lambda: Blocca_riconoscimento_voce(self, avvia_ric, indietro))

        avvia_ric = Button(self, width=10, bg="white", text="Start", command=lambda: Riconoscimento_vocale(self, blocca_registrazione, avvia_ric, indietro, risultati))

        risultati = Listbox(self, width=60, height=17, background="white")
        risultati.insert(END, "Results will be shown here...")
        risultati.insert(END, "\n\n")

        label_benvenuto.grid(row=0, column=0, sticky="N", pady=10, padx=65)
        avvia_ric.grid(row=1, column=0, sticky="W", pady=10, padx=200)
        indietro.grid(row=1, column=0, sticky="E", pady=10, padx=200)
        risultati.grid(row=2, column=0, sticky="N", pady=30)
def Pulisci_finestra_voce(self, avvia_ric, indietro):
    print("DENTRO PULISCI FINESTRA.")
    flag[0] = False

    avvia_ric.grid(row=1, column=0, sticky="W", pady=10, padx=200)
    indietro.grid(row=1, column=0, sticky="E", pady=10, padx=200)

    main_thread = threading.currentThread()
    # Elimino eventuali thread ancora in vita
    for t in threading.enumerate():
        if t is main_thread:
            continue
        print("Nome thread:", t.getName())
        t.join()
        print("Eliminato.")


    self.controller.show_frame("StartPage")

def load_wav(filename, sample_rate, random_crop):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)

    # Calculate its duration in seconds
    duration = int(librosa.get_duration(y=audio, sr=sr))  # duration in seconds

    return duration
def Riconoscimento_vocale(self, blocca_registrazione, avvia_ric, indietro, risultati):
    def Esecuzione_riconoscimento(self, blocca_registrazione, avvia_ric, indietro):
        file_report.write("\nEmotions Recognition through Voice Analysis\n")
        file_report.write(time.strftime("Start Time: %H:%M:%S\n\n"))
        class Thread_registra(Thread):
            def __init__(self, var):
                Thread.__init__(self)
                self.var = var

            def run(self):
                #label_emozione_riconosciuta['text'] = "Predizione in corso.."
                while (flag[0] == True):
                    time.sleep(2)
                    # Il thread principale si preoccuperà di continuare sempre a registrare l'audio. Ogni audio si interrompe quando la persona
                    # smette di parlare, appena questo succede il thread principale creerà e manderà in esecuzione il thread secondario che eseguirà
                    # altre operazioni descritte sotto. Subito dopo aver avviato il thread secondario il thread principale si rimette in ascolto e così via.
                    # record(c.FILE_TEST_DIR + str(self.var) + 'test.wav')  #################################################
                    f_audio.microfono(str(self.var))  # viene chiamata questa funzione che attiva il microfono e inizia a registrare fino a quando l'utente parla.
                    # L'audio che viene registrato viene salvato sul file chiamato "audio_file_NumeroFile.wav". La parte NumeroFile è utile perchè
                    # permette di creare per ogni registrazione un file diverso in modo tale che il thread principale non salvi sempre audio differenti
                    # su uniìo stesso file audio.

                    t_secondario = Thread_secondario(str(self.var), self.var)
                    t_secondario.start()
                    self.var = self.var + 1

        class Thread_secondario(Thread):
            # FUNZIONAMENTO THREAD SECONDARIO: ogni thread secondario chiama prima la funziona per eliminare il silenzio dal file audio che gli è stato
            # affidato, successivamente esegue delle funzioni sul file per estrarre le features necessarie, dopodichè fornisce in input al modello tali features
            # e dopo che il modello ha dato la sua risposta, il thread la stampa su un'etichetta; infine sempre il thread elimina il file audio affidatogli
            # e termina la sua esecuzione.

            def __init__(self, file_test, var):
                Thread.__init__(self)
                self.file_test = file_test
                self.var = var

            def run(self):
                if (flag[0] == True):
                    # Passo 1-eliminazione silenzio:
                    #######RIMUOVO IL SILENZIO DALL'AUDIO###################
                    nome_file_audio_del_thread_corrente = 'audio_file' + str(self.var) + '.wav'

                    # sound = AudioSegment.from_file('C:\\Users\\win 10\\PycharmProjects\\Programmi\\audio_file.wav',format="wav")
                    sound = AudioSegment.from_file(nome_file_audio_del_thread_corrente,format="wav")

                    start_trim = f_audio.detect_leading_silence(sound)
                    end_trim = f_audio.detect_leading_silence(sound.reverse())

                    duration = len(sound)
                    trimmed_sound = sound[start_trim:duration - end_trim]

                    # il file di output che conterrà l'audio senza silenzio è sempre quello di prima in modo da non creare troppi file ogni volta.
                    trimmed_sound.export(nome_file_audio_del_thread_corrente,format="wav")

                    # il file di output che conterrà l'audio senza silenzio è sempre quello di prima in modo da non creare troppi file ogni volta.
                    # trimmed_sound.export('C:\\Users\\win 10\\PycharmProjects\\Programmi\\' +nome_file_audio_output_del_thread_corrente,format="wav")
                    ######################################################## ELIMINAZIONE SILENZIO TERMINATA

                    #DEVI METTERE LA FUNZIONE CHE SE L'AUDIO DURA MENO DI 1 SECONDO VIENE SCARTATO
                    durata = load_wav(nome_file_audio_del_thread_corrente, 16000, True)

                    if (durata != 0):

                        # Passo 2-estrazione features: #PER L'AROUSAL (SVM)
                        # self.filename_iniziale = nome_file_audio_del_thread_corrente
                        self.filename = nome_file_audio_del_thread_corrente
                        # print("Thread numero  " +str(self.var ) +" ha il file  " +self.filename)

                        y, sr = librosa.load(self.filename,sr=16000)  # librosa estrae delle features sr = lista  delle features e y = lunghezza della lista delle features

                        # ESTRAGGO SUBITO LO SPETTROGRAMMA DAL FILE AUDIO:
                        ps = librosa.feature.melspectrogram(y=y, sr=sr)
                        # print("ps.shape prima: ", ps.shape)

                        if (ps.shape[0] != 128 or ps.shape[1] != 128):  # se lo spettrogramma non è della dimension (128,128)
                            # allora con questa piccola parte di codice lo riporto a tale dimensione con la tecnica dello
                            # zero-padding:
                            ps_temp = np.zeros([128, 128])

                            if (ps.shape[1] <= 128):
                                for i in range(0, 128):
                                    for j in range(0, ps.shape[1]):
                                        ps_temp[i][j] = ps[i][j]
                                ps = ps_temp
                            else:
                                for i in range(0, 128):
                                    for j in range(0, 128):
                                        ps_temp[i][j] = ps[i][j]
                                ps = ps_temp
                        # TECNICA ZERO-PADDING COMPLETATA (qualora fosse stata applicata)
                        # trasformo l'input per darlo alla rete neurale:
                        ps = ps.reshape(1, 128, 128, 1)
                        # print("ps.shape dopo averlo trasformato: ", ps.shape)

                        # A QUESTO PUNTO PARTE L'ESTRAZIONE DELLE FEATURES PER LA SVM:
                        y2 = librosa.effects.harmonic(y)
                        S = np.abs(librosa.stft(y))
                        mfcc_vector = librosa.feature.mfcc(y=y, sr=sr)
                        delta_vector = librosa.feature.delta(mfcc_vector)
                        delta_delta_vector = librosa.feature.delta(mfcc_vector, order=2)

                        # calcolo la frequenza fondamentale:
                        pitch, magnitudes = librosa.piptrack(y=y, sr=sr)
                        media_pitch = np.mean(pitch)
                        min_pitch = np.min(pitch)
                        max_pitch = np.max(pitch)
                        media_log_pitch = np.log(media_pitch)
                        max_log_pitch = np.log(max_pitch)
                        dev_std_pitch = np.std(pitch)
                        # estraggo features per zero crossing rate:
                        zero_c_r = librosa.feature.zero_crossing_rate(y)
                        media_zero_c_r = np.mean(zero_c_r)
                        min_z_c_r = np.min(zero_c_r)
                        max_z_c_r = np.max(zero_c_r)
                        dev_std_z_c_r = np.std(zero_c_r)

                        # estraggo le features dell'energia:
                        S, phase = librosa.magphase(librosa.stft(y))
                        rms = librosa.feature.rms(S=S)
                        media_rms = np.mean(rms)
                        min_rms = np.min(rms)
                        max_rms = np.max(rms)
                        dev_std_rms = np.std(rms)
                        rms_mediana = np.median(rms)

                        vett_features = []  # ogni volta devo creare un nuovo vettore che conterrà le features solo dell'audio corrente

                        media_c_0 = np.mean(mfcc_vector[0])
                        mediana_c_0 = np.median(mfcc_vector[0])
                        dev_stand_c_0 = np.std(mfcc_vector[0])

                        delta_media_c_0 = np.mean(delta_vector[0])
                        delta_mediana_c_0 = np.median(delta_vector[0])
                        delta_dev_stand_c_0 = np.std(delta_vector[0])

                        delta_delta_media_c_0 = np.mean(delta_delta_vector[0])
                        delta_delta_mediana_c_0 = np.median(delta_delta_vector[0])
                        delta_delta_dev_stand_c_0 = np.std(delta_delta_vector[0])

                        media_c_1 = np.mean(mfcc_vector[1])
                        mediana_c_1 = np.median(mfcc_vector[1])
                        dev_stand_c_1 = np.std(mfcc_vector[1])

                        delta_media_c_1 = np.mean(delta_vector[1])
                        delta_mediana_c_1 = np.median(delta_vector[1])
                        delta_dev_stand_c_1 = np.std(delta_vector[1])

                        delta_delta_media_c_1 = np.mean(delta_delta_vector[1])
                        delta_delta_mediana_c_1 = np.median(delta_delta_vector[1])
                        delta_delta_dev_stand_c_1 = np.std(delta_delta_vector[1])

                        media_c_2 = np.mean(mfcc_vector[2])
                        mediana_c_2 = np.median(mfcc_vector[2])
                        dev_stand_c_2 = np.std(mfcc_vector[2])

                        delta_media_c_2 = np.mean(delta_vector[2])
                        delta_mediana_c_2 = np.median(delta_vector[2])
                        delta_dev_stand_c_2 = np.std(delta_vector[2])

                        delta_delta_media_c_2 = np.mean(delta_delta_vector[2])
                        delta_delta_mediana_c_2 = np.median(delta_delta_vector[2])
                        delta_delta_dev_stand_c_2 = np.std(delta_delta_vector[2])

                        media_c_3 = np.mean(mfcc_vector[3])
                        mediana_c_3 = np.median(mfcc_vector[3])
                        dev_stand_c_3 = np.std(mfcc_vector[3])

                        delta_media_c_3 = np.mean(delta_vector[3])
                        delta_mediana_c_3 = np.median(delta_vector[3])
                        delta_dev_stand_c_3 = np.std(delta_vector[3])

                        delta_delta_media_c_3 = np.mean(delta_delta_vector[3])
                        delta_delta_mediana_c_3 = np.median(delta_delta_vector[3])
                        delta_delta_dev_stand_c_3 = np.std(delta_delta_vector[3])

                        media_c_4 = np.mean(mfcc_vector[4])
                        mediana_c_4 = np.median(mfcc_vector[4])
                        dev_stand_c_4 = np.std(mfcc_vector[4])

                        delta_media_c_4 = np.mean(delta_vector[4])
                        delta_mediana_c_4 = np.median(delta_vector[4])
                        delta_dev_stand_c_4 = np.std(delta_vector[4])

                        delta_delta_media_c_4 = np.mean(delta_delta_vector[4])
                        delta_delta_mediana_c_4 = np.median(delta_delta_vector[4])
                        delta_delta_dev_stand_c_4 = np.std(delta_delta_vector[4])

                        media_c_5 = np.mean(mfcc_vector[5])
                        mediana_c_5 = np.median(mfcc_vector[5])
                        dev_stand_c_5 = np.std(mfcc_vector[5])

                        delta_media_c_5 = np.mean(delta_vector[5])
                        delta_mediana_c_5 = np.median(delta_vector[5])
                        delta_dev_stand_c_5 = np.std(delta_vector[5])

                        delta_delta_media_c_5 = np.mean(delta_delta_vector[5])
                        delta_delta_mediana_c_5 = np.median(delta_delta_vector[5])
                        delta_delta_dev_stand_c_5 = np.std(delta_delta_vector[5])

                        media_c_6 = np.mean(mfcc_vector[6])
                        mediana_c_6 = np.median(mfcc_vector[6])
                        dev_stand_c_6 = np.std(mfcc_vector[6])

                        delta_media_c_6 = np.mean(delta_vector[6])
                        delta_mediana_c_6 = np.median(delta_vector[6])
                        delta_dev_stand_c_6 = np.std(delta_vector[6])

                        delta_delta_media_c_6 = np.mean(delta_delta_vector[6])
                        delta_delta_mediana_c_6 = np.median(delta_delta_vector[6])
                        delta_delta_dev_stand_c_6 = np.std(delta_delta_vector[6])

                        media_c_7 = np.mean(mfcc_vector[7])
                        mediana_c_7 = np.median(mfcc_vector[7])
                        dev_stand_c_7 = np.std(mfcc_vector[7])

                        delta_media_c_7 = np.mean(delta_vector[7])
                        delta_mediana_c_7 = np.median(delta_vector[7])
                        delta_dev_stand_c_7 = np.std(delta_vector[7])

                        delta_delta_media_c_7 = np.mean(delta_delta_vector[7])
                        delta_delta_mediana_c_7 = np.median(delta_delta_vector[7])
                        delta_delta_dev_stand_c_7 = np.std(delta_delta_vector[7])

                        media_c_8 = np.mean(mfcc_vector[8])
                        mediana_c_8 = np.median(mfcc_vector[8])
                        dev_stand_c_8 = np.std(mfcc_vector[8])

                        delta_media_c_8 = np.mean(delta_vector[8])
                        delta_mediana_c_8 = np.median(delta_vector[8])
                        delta_dev_stand_c_8 = np.std(delta_vector[8])

                        delta_delta_media_c_8 = np.mean(delta_delta_vector[8])
                        delta_delta_mediana_c_8 = np.median(delta_delta_vector[8])
                        delta_delta_dev_stand_c_8 = np.std(delta_delta_vector[8])

                        media_c_9 = np.mean(mfcc_vector[9])
                        mediana_c_9 = np.median(mfcc_vector[9])
                        dev_stand_c_9 = np.std(mfcc_vector[9])

                        delta_media_c_9 = np.mean(delta_vector[9])
                        delta_mediana_c_9 = np.median(delta_vector[9])
                        delta_dev_stand_c_9 = np.std(delta_vector[9])

                        delta_delta_media_c_9 = np.mean(delta_delta_vector[9])
                        delta_delta_mediana_c_9 = np.median(delta_delta_vector[9])
                        delta_delta_dev_stand_c_9 = np.std(delta_delta_vector[9])

                        media_c_10 = np.mean(mfcc_vector[10])
                        mediana_c_10 = np.median(mfcc_vector[10])
                        dev_stand_c_10 = np.std(mfcc_vector[10])

                        delta_media_c_10 = np.mean(delta_vector[10])
                        delta_mediana_c_10 = np.median(delta_vector[10])
                        delta_dev_stand_c_10 = np.std(delta_vector[10])

                        delta_delta_media_c_10 = np.mean(delta_delta_vector[10])
                        delta_delta_mediana_c_10 = np.median(delta_delta_vector[10])
                        delta_delta_dev_stand_c_10 = np.std(delta_delta_vector[10])

                        media_c_11 = np.mean(mfcc_vector[11])
                        mediana_c_11 = np.median(mfcc_vector[11])
                        dev_stand_c_11 = np.std(mfcc_vector[11])

                        delta_media_c_11 = np.mean(delta_vector[11])
                        delta_mediana_c_11 = np.median(delta_vector[11])
                        delta_dev_stand_c_11 = np.std(delta_vector[11])

                        delta_delta_media_c_11 = np.mean(delta_delta_vector[11])
                        delta_delta_mediana_c_11 = np.median(delta_delta_vector[11])
                        delta_delta_dev_stand_c_11 = np.std(delta_delta_vector[11])

                        media_c_12 = np.mean(mfcc_vector[12])
                        mediana_c_12 = np.median(mfcc_vector[12])
                        dev_stand_c_12 = np.std(mfcc_vector[12])

                        delta_media_c_12 = np.mean(delta_vector[12])
                        delta_mediana_c_12 = np.median(delta_vector[12])
                        delta_dev_stand_c_12 = np.std(delta_vector[12])

                        delta_delta_media_c_12 = np.mean(delta_delta_vector[12])
                        delta_delta_mediana_c_12 = np.median(delta_delta_vector[12])
                        delta_delta_dev_stand_c_12 = np.std(delta_delta_vector[12])

                        media_c_13 = np.mean(mfcc_vector[13])
                        mediana_c_13 = np.median(mfcc_vector[13])
                        dev_stand_c_13 = np.std(mfcc_vector[13])

                        delta_media_c_13 = np.mean(delta_vector[13])
                        delta_mediana_c_13 = np.median(delta_vector[13])
                        delta_dev_stand_c_13 = np.std(delta_vector[13])

                        delta_delta_media_c_13 = np.mean(delta_delta_vector[13])
                        delta_delta_mediana_c_13 = np.median(delta_delta_vector[13])
                        delta_delta_dev_stand_c_13 = np.std(delta_delta_vector[13])

                        media_c_14 = np.mean(mfcc_vector[14])
                        mediana_c_14 = np.median(mfcc_vector[14])
                        dev_stand_c_14 = np.std(mfcc_vector[14])

                        delta_media_c_14 = np.mean(delta_vector[14])
                        delta_mediana_c_14 = np.median(delta_vector[14])
                        delta_dev_stand_c_14 = np.std(delta_vector[14])

                        delta_delta_media_c_14 = np.mean(delta_delta_vector[14])
                        delta_delta_mediana_c_14 = np.median(delta_delta_vector[14])
                        delta_delta_dev_stand_c_14 = np.std(delta_delta_vector[14])

                        media_c_15 = np.mean(mfcc_vector[15])
                        mediana_c_15 = np.median(mfcc_vector[15])
                        dev_stand_c_15 = np.std(mfcc_vector[15])

                        delta_media_c_15 = np.mean(delta_vector[15])
                        delta_mediana_c_15 = np.median(delta_vector[15])
                        delta_dev_stand_c_15 = np.std(delta_vector[15])

                        delta_delta_media_c_15 = np.mean(delta_delta_vector[15])
                        delta_delta_mediana_c_15 = np.median(delta_delta_vector[15])
                        delta_delta_dev_stand_c_15 = np.std(delta_delta_vector[15])

                        media_c_16 = np.mean(mfcc_vector[16])
                        mediana_c_16 = np.median(mfcc_vector[16])
                        dev_stand_c_16 = np.std(mfcc_vector[16])

                        delta_media_c_16 = np.mean(delta_vector[16])
                        delta_mediana_c_16 = np.median(delta_vector[16])
                        delta_dev_stand_c_16 = np.std(delta_vector[16])

                        delta_delta_media_c_16 = np.mean(delta_delta_vector[16])
                        delta_delta_mediana_c_16 = np.median(delta_delta_vector[16])
                        delta_delta_dev_stand_c_16 = np.std(delta_delta_vector[16])

                        media_c_17 = np.mean(mfcc_vector[17])
                        mediana_c_17 = np.median(mfcc_vector[17])
                        dev_stand_c_17 = np.std(mfcc_vector[17])

                        delta_media_c_17 = np.mean(delta_vector[17])
                        delta_mediana_c_17 = np.median(delta_vector[17])
                        delta_dev_stand_c_17 = np.std(delta_vector[17])

                        delta_delta_media_c_17 = np.mean(delta_delta_vector[17])
                        delta_delta_mediana_c_17 = np.median(delta_delta_vector[17])
                        delta_delta_dev_stand_c_17 = np.std(delta_delta_vector[17])

                        media_c_18 = np.mean(mfcc_vector[18])
                        mediana_c_18 = np.median(mfcc_vector[18])
                        dev_stand_c_18 = np.std(mfcc_vector[18])

                        delta_media_c_18 = np.mean(delta_vector[18])
                        delta_mediana_c_18 = np.median(delta_vector[18])
                        delta_dev_stand_c_18 = np.std(delta_vector[18])

                        delta_delta_media_c_18 = np.mean(delta_delta_vector[18])
                        delta_delta_mediana_c_18 = np.median(delta_delta_vector[18])
                        delta_delta_dev_stand_c_18 = np.std(delta_delta_vector[18])

                        media_c_19 = np.mean(mfcc_vector[19])
                        mediana_c_19 = np.median(mfcc_vector[19])
                        dev_stand_c_19 = np.std(mfcc_vector[19])

                        delta_media_c_19 = np.mean(delta_vector[19])
                        delta_mediana_c_19 = np.median(delta_vector[19])
                        delta_dev_stand_c_19 = np.std(delta_vector[19])

                        delta_delta_media_c_19 = np.mean(delta_delta_vector[19])
                        delta_delta_mediana_c_19 = np.median(delta_delta_vector[19])
                        delta_delta_dev_stand_c_19 = np.std(delta_delta_vector[19])

                        # metto insieme tutte le features estratte in modo da creare un unico vettore da 180 elementi:
                        vett_features.append(media_c_0)
                        vett_features.append(mediana_c_0)
                        vett_features.append(dev_stand_c_0)

                        vett_features.append(delta_media_c_0)
                        vett_features.append(delta_mediana_c_0)
                        vett_features.append(delta_dev_stand_c_0)

                        vett_features.append(delta_delta_media_c_0)
                        vett_features.append(delta_delta_mediana_c_0)
                        vett_features.append(delta_delta_dev_stand_c_0)

                        vett_features.append(media_c_1)
                        vett_features.append(mediana_c_1)
                        vett_features.append(dev_stand_c_1)

                        vett_features.append(delta_media_c_1)
                        vett_features.append(delta_mediana_c_1)
                        vett_features.append(delta_dev_stand_c_1)

                        vett_features.append(delta_delta_media_c_1)
                        vett_features.append(delta_delta_mediana_c_1)
                        vett_features.append(delta_delta_dev_stand_c_1)

                        vett_features.append(media_c_2)
                        vett_features.append(mediana_c_2)
                        vett_features.append(dev_stand_c_2)

                        vett_features.append(delta_media_c_2)
                        vett_features.append(delta_mediana_c_2)
                        vett_features.append(delta_dev_stand_c_2)

                        vett_features.append(delta_delta_media_c_2)
                        vett_features.append(delta_delta_mediana_c_2)
                        vett_features.append(delta_delta_dev_stand_c_2)

                        vett_features.append(media_c_3)
                        vett_features.append(mediana_c_3)
                        vett_features.append(dev_stand_c_3)

                        vett_features.append(delta_media_c_3)
                        vett_features.append(delta_mediana_c_3)
                        vett_features.append(delta_dev_stand_c_3)

                        vett_features.append(delta_delta_media_c_3)
                        vett_features.append(delta_delta_mediana_c_3)
                        vett_features.append(delta_delta_dev_stand_c_3)

                        vett_features.append(media_c_4)
                        vett_features.append(mediana_c_4)
                        vett_features.append(dev_stand_c_4)

                        vett_features.append(delta_media_c_4)
                        vett_features.append(delta_mediana_c_4)
                        vett_features.append(delta_dev_stand_c_4)

                        vett_features.append(delta_delta_media_c_4)
                        vett_features.append(delta_delta_mediana_c_4)
                        vett_features.append(delta_delta_dev_stand_c_4)

                        vett_features.append(media_c_5)
                        vett_features.append(mediana_c_5)
                        vett_features.append(dev_stand_c_5)

                        vett_features.append(delta_media_c_5)
                        vett_features.append(delta_mediana_c_5)
                        vett_features.append(delta_dev_stand_c_5)

                        vett_features.append(delta_delta_media_c_5)
                        vett_features.append(delta_delta_mediana_c_5)
                        vett_features.append(delta_delta_dev_stand_c_5)

                        vett_features.append(media_c_6)
                        vett_features.append(mediana_c_6)
                        vett_features.append(dev_stand_c_6)

                        vett_features.append(delta_media_c_6)
                        vett_features.append(delta_mediana_c_6)
                        vett_features.append(delta_dev_stand_c_6)

                        vett_features.append(delta_delta_media_c_6)
                        vett_features.append(delta_delta_mediana_c_6)
                        vett_features.append(delta_delta_dev_stand_c_6)

                        vett_features.append(media_c_7)
                        vett_features.append(mediana_c_7)
                        vett_features.append(dev_stand_c_7)

                        vett_features.append(delta_media_c_7)
                        vett_features.append(delta_mediana_c_7)
                        vett_features.append(delta_dev_stand_c_7)

                        vett_features.append(delta_delta_media_c_7)
                        vett_features.append(delta_delta_mediana_c_7)
                        vett_features.append(delta_delta_dev_stand_c_7)

                        vett_features.append(media_c_8)
                        vett_features.append(mediana_c_8)
                        vett_features.append(dev_stand_c_8)

                        vett_features.append(delta_media_c_8)
                        vett_features.append(delta_mediana_c_8)
                        vett_features.append(delta_dev_stand_c_8)

                        vett_features.append(delta_delta_media_c_8)
                        vett_features.append(delta_delta_mediana_c_8)
                        vett_features.append(delta_delta_dev_stand_c_8)

                        vett_features.append(media_c_9)
                        vett_features.append(mediana_c_9)
                        vett_features.append(dev_stand_c_9)

                        vett_features.append(delta_media_c_9)
                        vett_features.append(delta_mediana_c_9)
                        vett_features.append(delta_dev_stand_c_9)

                        vett_features.append(delta_delta_media_c_9)
                        vett_features.append(delta_delta_mediana_c_9)
                        vett_features.append(delta_delta_dev_stand_c_9)

                        vett_features.append(media_c_10)
                        vett_features.append(mediana_c_10)
                        vett_features.append(dev_stand_c_10)

                        vett_features.append(delta_media_c_10)
                        vett_features.append(delta_mediana_c_10)
                        vett_features.append(delta_dev_stand_c_10)

                        vett_features.append(delta_delta_media_c_10)
                        vett_features.append(delta_delta_mediana_c_10)
                        vett_features.append(delta_delta_dev_stand_c_10)

                        vett_features.append(media_c_11)
                        vett_features.append(mediana_c_11)
                        vett_features.append(dev_stand_c_11)

                        vett_features.append(delta_media_c_11)
                        vett_features.append(delta_mediana_c_11)
                        vett_features.append(delta_dev_stand_c_11)

                        vett_features.append(delta_delta_media_c_11)
                        vett_features.append(delta_delta_mediana_c_11)
                        vett_features.append(delta_delta_dev_stand_c_11)

                        vett_features.append(media_c_12)
                        vett_features.append(mediana_c_12)
                        vett_features.append(dev_stand_c_12)

                        vett_features.append(delta_media_c_12)
                        vett_features.append(delta_mediana_c_12)
                        vett_features.append(delta_dev_stand_c_12)

                        vett_features.append(delta_delta_media_c_12)
                        vett_features.append(delta_delta_mediana_c_12)
                        vett_features.append(delta_delta_dev_stand_c_12)

                        vett_features.append(media_c_13)
                        vett_features.append(mediana_c_13)
                        vett_features.append(dev_stand_c_13)

                        vett_features.append(delta_media_c_13)
                        vett_features.append(delta_mediana_c_13)
                        vett_features.append(delta_dev_stand_c_13)

                        vett_features.append(delta_delta_media_c_13)
                        vett_features.append(delta_delta_mediana_c_13)
                        vett_features.append(delta_delta_dev_stand_c_13)

                        vett_features.append(media_c_14)
                        vett_features.append(mediana_c_14)
                        vett_features.append(dev_stand_c_14)

                        vett_features.append(delta_media_c_14)
                        vett_features.append(delta_mediana_c_14)
                        vett_features.append(delta_dev_stand_c_14)

                        vett_features.append(delta_delta_media_c_14)
                        vett_features.append(delta_delta_mediana_c_14)
                        vett_features.append(delta_delta_dev_stand_c_14)

                        vett_features.append(media_c_15)
                        vett_features.append(mediana_c_15)
                        vett_features.append(dev_stand_c_15)

                        vett_features.append(delta_media_c_15)
                        vett_features.append(delta_mediana_c_15)
                        vett_features.append(delta_dev_stand_c_15)

                        vett_features.append(delta_delta_media_c_15)
                        vett_features.append(delta_delta_mediana_c_15)
                        vett_features.append(delta_delta_dev_stand_c_15)

                        vett_features.append(media_c_16)
                        vett_features.append(mediana_c_16)
                        vett_features.append(dev_stand_c_16)

                        vett_features.append(delta_media_c_16)
                        vett_features.append(delta_mediana_c_16)
                        vett_features.append(delta_dev_stand_c_16)

                        vett_features.append(delta_delta_media_c_16)
                        vett_features.append(delta_delta_mediana_c_16)
                        vett_features.append(delta_delta_dev_stand_c_16)

                        vett_features.append(media_c_17)
                        vett_features.append(mediana_c_17)
                        vett_features.append(dev_stand_c_17)

                        vett_features.append(delta_media_c_17)
                        vett_features.append(delta_mediana_c_17)
                        vett_features.append(delta_dev_stand_c_17)

                        vett_features.append(delta_delta_media_c_17)
                        vett_features.append(delta_delta_mediana_c_17)
                        vett_features.append(delta_delta_dev_stand_c_17)

                        vett_features.append(media_c_18)
                        vett_features.append(mediana_c_18)
                        vett_features.append(dev_stand_c_18)

                        vett_features.append(delta_media_c_18)
                        vett_features.append(delta_mediana_c_18)
                        vett_features.append(delta_dev_stand_c_18)

                        vett_features.append(delta_delta_media_c_18)
                        vett_features.append(delta_delta_mediana_c_18)
                        vett_features.append(delta_delta_dev_stand_c_18)

                        vett_features.append(media_c_19)
                        vett_features.append(mediana_c_19)
                        vett_features.append(dev_stand_c_19)

                        vett_features.append(delta_media_c_19)
                        vett_features.append(delta_mediana_c_19)
                        vett_features.append(delta_dev_stand_c_19)

                        vett_features.append(delta_delta_media_c_19)
                        vett_features.append(delta_delta_mediana_c_19)
                        vett_features.append(delta_delta_dev_stand_c_19)

                        vett_features.append(media_pitch)
                        vett_features.append(min_pitch)
                        vett_features.append(max_pitch)
                        vett_features.append(media_log_pitch)
                        vett_features.append(max_log_pitch)
                        vett_features.append(dev_std_pitch)

                        vett_features.append(media_zero_c_r)
                        vett_features.append(min_z_c_r)
                        vett_features.append(max_z_c_r)
                        vett_features.append(dev_std_z_c_r)

                        vett_features.append(media_rms)
                        vett_features.append(min_rms)
                        vett_features.append(max_rms)
                        vett_features.append(dev_std_rms)
                        vett_features.append(rms_mediana)
                        # TERMINA L'ESTRAZIONE DELLE FEATURES

                        # Passo 3 - acquisizione oggetti per eseguire la predizione:
                        self.model = modello_caricato  # il thread secondario corrente prende l'istanza del modello della SVM per effettuare la predizione
                        self.scaler = scaler  # prende l'oggetto scaler

                        self.modello_SVM_GIOvsSOR = modello_svm_gioia_vs_sorpresa
                        self.scaler_SVM_GIOvsSOR = scaler_svm_gioia_vs_sorpresa
                        self.modello_SVM_PAUvsRAB = modello_svm_paura_vs_rabbia
                        self.scaler_SVM_PAUvsRAB = scaler_svm_paura_vs_rabbia

                        self.modello_CNN = loaded_model  # il thread secondario corrente prende l'istanza del modello della CNN per effettuare la predizione
                        ###########################################################

                        # Passo4 - fornire in input il vettore delle features al modello e ottenere la predizione:
                        vett_features_one_vs_one = []
                        vett_features_one_vs_one = vett_features

                        vett_features = np.expand_dims(vett_features, axis=0)  # cambio la dim al vettore
                        vett_features = self.scaler.transform(vett_features)  # scalo il vettore

                        # effettuo la predizione
                        # print("Pred prob - AROUSAL: ", modello_caricato.predict_proba(vett_features))
                        prediction = self.model.predict(vett_features)
                        # print("Pred audio - AROUSAL:", prediction)

                        # MEMORIZZO LA PREDIZIONE CHE MI HA DATO LA SVM SULL'AROUSAL
                        #print("VALORE AROUSAL: ", prediction[0])
                        self.predizione_arousal = prediction[0]  # In questa variabile mi conservo la predizione della SVM sull'arousal

                        # ORA UTILIZZO LA CNN PER PREDIRE LA VALENCE DELL'AUDIO:
                        prob_valence = self.modello_CNN.predict_proba(ps)
                        self.predizione_valence = self.modello_CNN.predict_classes(ps)
                        # print("Pred prob - VALENCE::", prob_valence)
                        #print("Pred audio - VALENCE:: ", self.predizione_valence)

                        if (self.predizione_valence == 2 and self.predizione_arousal == 3):
                            print("VALENCE: NEGATIVA")
                            print("AROUSAL: NEGATIVA")
                            print("PRED: DISGUSTO")
                            self.predizione = 'Disgusto'
                            risultati.insert(END, "Recognized emotion: disgust\n")
                            risultati.insert(END, "Valence: negative\n")
                            risultati.insert(END, "Arousal: negative\n")
                            risultati.insert(END, "\n")
                            file_report.write("Recognized emotion: disgust\n")
                            file_report.write("Valence: negative\n")
                            file_report.write("Arousal: negative\n")
                            file_report.write("\n")

                        elif (self.predizione_valence == 0 and self.predizione_arousal == 1):
                            vett_features_one_vs_one = np.expand_dims(vett_features_one_vs_one, axis=0)
                            vett_features_one_vs_one = self.scaler_SVM_GIOvsSOR.transform(
                                vett_features_one_vs_one)  # scalo il vettore
                            prediction = modello_svm_gioia_vs_sorpresa.predict(vett_features_one_vs_one)
                            if (prediction[0] == 0):
                                print("VALENCE: POSITIVA")
                                print("AROUSAL: ALTA")
                                print("PRED: GIOIA")
                                self.predizione = "Gioia"
                                risultati.insert(END, "Recognized emotion: happy")
                                risultati.insert(END, "Valence: positive")
                                risultati.insert(END, "Arousal: high")
                                risultati.insert(END, "\n")
                                file_report.write("Recognized emotion: happy\n")
                                file_report.write("Valence: positive\n")
                                file_report.write("Arousal: high\n")
                                file_report.write("\n")
                            else:
                                print("VALENCE: POSITIVA")
                                print("AROUSAL: ALTA")
                                print("PRED: SORPRESA")
                                self.predizione = "Sorpresa"
                                risultati.insert(END, "Recognized emotion: surprise")
                                risultati.insert(END, "Valence: positive")
                                risultati.insert(END, "Arousal: high")
                                risultati.insert(END, "\n")
                                file_report.write("Recognized emotion: surprise\n")
                                file_report.write("Valence: positive\n")
                                file_report.write("Arousal: high\n")
                                file_report.write("\n")

                        elif (self.predizione_valence == 1 and self.predizione_arousal == 2):
                            print("VALENCE: NEUTRALE")
                            print("AROUSAL: MEDIA")
                            print("PRED: NEUTRALE")
                            self.predizione = 'Neutrale'
                            risultati.insert(END, "Recognized emotion: neutral")
                            risultati.insert(END, "Valence: neutral")
                            risultati.insert(END, "Arousal: medium")
                            risultati.insert(END, "\n")
                            file_report.write("Recognized emotion: neutral\n")
                            file_report.write("Valence: neutral\n")
                            file_report.write("Arousal: medium\n")
                            file_report.write("\n")

                        elif (self.predizione_valence == 2 and self.predizione_arousal == 1):
                            vett_features_one_vs_one = np.expand_dims(vett_features_one_vs_one, axis=0)
                            vett_features_one_vs_one = self.scaler_SVM_PAUvsRAB.transform(
                                vett_features_one_vs_one)  # scalo il vettore
                            prediction = modello_svm_paura_vs_rabbia.predict(vett_features_one_vs_one)
                            if (prediction[0] == 0):
                                print("VALENCE: NEGATIVA")
                                print("AROUSAL: ALTA")
                                print("PRED: PAURA")
                                self.predizione = "Paura"
                                risultati.insert(END, "Recognized emotion: fear")
                                risultati.insert(END, "Valence: negative")
                                risultati.insert(END, "Arousal: high")
                                risultati.insert(END, "\n")
                                file_report.write("Recognized emotion: fear\n")
                                file_report.write("Valence: negative\n")
                                file_report.write("Arousal: high\n")
                                file_report.write("\n")
                            else:
                                print("VALENCE: NEGATIVA")
                                print("AROUSAL: ALTA")
                                print("PRED: RABBIA")
                                self.predizione = "Rabbia"
                                risultati.insert(END, "Recognized emotion: anger")
                                risultati.insert(END, "Valence: negative")
                                risultati.insert(END, "Arousal: high")
                                risultati.insert(END, "\n")
                                file_report.write("Recognized emotion: anger\n")
                                file_report.write("Valence: negative\n")
                                file_report.write("Arousal: high\n")
                                file_report.write("\n")

                        elif (self.predizione_valence == 2 and self.predizione_arousal == 2):
                            print("VALENCE: NEGATIVA")
                            print("AROUSAL: MEDIA")
                            print("PRED: TRISTEZZA")
                            self.predizione = 'Tristezza'
                            risultati.insert(END, "Recognized emotion: sadness")
                            risultati.insert(END, "Valence: negative")
                            risultati.insert(END, "Arousal: medium")
                            risultati.insert(END, "\n")
                            file_report.write("Recognized emotion: sadness\n")
                            file_report.write("Valence: negative\n")
                            file_report.write("Arousal: medium\n")
                            file_report.write("\n")

                        else:
                            print("PRED: NESSUNA COMBINAZIONE")
                            self.predizione = 'Nessuna predizione'
                            risultati.insert(END, "Recognized emotion: -")
                            risultati.insert(END, "\n")
                            file_report.write("Recognized emotion: -\n")
                            file_report.write("\n")

                        if (flag[0] == True):
                            print(" ")

                        os.unlink(self.filename)


                    else:
                        print("IL FILE ERA TROPPO CORTO (INFERIORE AD UN SECONDO) E QUINDI NON E' STATO CONSIDERATO.")

        var0 = 1

        avvia_ric.grid_forget()
        blocca_registrazione.grid(row=1, column=0, sticky='w', pady=10, padx=200)




        self.update()
        t_registra = Thread_registra(var0)
        t_registra.start()

    ############################################################################ FINE FUNZIONE chiamata Riconoscimento

    flag[0] = True

    indietro.grid(row=1, column=0, sticky='e', pady=10, padx=200)  #posiziono il pulsante indietro sotto la listbox
    # attach listbox to scrollbar
    #listbox.config(yscrollcommand=scrollbar.set)
    # scrollbar.config(command=listbox.yview)

    # label.pack()

    Esecuzione_riconoscimento(self, blocca_registrazione, avvia_ric, indietro)
def Blocca_riconoscimento_voce(self, avvia_ric, indietro):
    print("DENTRO PULISCI FINESTRA.")
    flag[0] = False

    avvia_ric.grid(row=1, column=0, sticky="W", pady=10, padx=200)
    indietro.grid(row=1, column=0, sticky="E", pady=10, padx=200)

    main_thread = threading.currentThread()
    tkm.showinfo("Warning", message="Vocal recognition terminated!")
    # Elimino eventuali thread ancora in vita
    for t in threading.enumerate():
        if t is main_thread:
            continue
        print("Nome thread:", t.getName())
        t.join()
        print("Eliminato.")


    file_report.write(time.strftime("End Time: %H:%M:%S\n"))
############################################################################
#                   FINE RICONOSCIMENTO SOLO DALLA VOCE                    #
############################################################################




############################################################################
#                   INIZIO RICONOSCIMENTO SOLO DAL VOLTO                   #
############################################################################

class Page_Volto(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(background="midnight blue")
        label_benvenuto = Label(self, width=50, height=3, font=('black', 13, 'bold'), bg="green yellow",  text="Welcome to Emotions Recognition through Face")

        indietro = tk.Button(self, width=10, background="white", text="Back",highlightbackground ="gray", activebackground="gray",command=lambda: Pulisci_finestra_volto(self))

        avvia_ric = Button(self, width=10, text="Start", highlightbackground="gray", activebackground="gray",
                              command=lambda: Esecuzione_riconoscimento_volto(self))
        info = Text(self, width=50, height=5, background="white")
        info.insert(INSERT, "Press Q on the keyboard to close 'My Face' window that will open...")

        label_benvenuto.grid(row=0, column=0, sticky="N", pady=10, padx=95)
        avvia_ric.grid(row=1, column=0, sticky="W", pady=10, padx=200)
        indietro.grid(row=1, column=0, sticky="E", pady=10, padx=200)
        info.grid(row=2, column=0, sticky="N", pady=30)
def Pulisci_finestra_volto(self):
    print("DENTRO PULISCI FINESTRA.")
    self.controller.show_frame("StartPage")
def Esecuzione_riconoscimento_volto(self):
    file_report.write("\n\nEmotions Recognition through Face Analysis\n")
    file_report.write(time.strftime("Start Time: %H:%M:%S\n\n"))
    # CARICAMENTO CLASSIFICATORI
    face_detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    emotion_classifier = load_model("emotions.hdf5", compile=False)
    EMOTIONS = ["angry", "disgusted", "scared", "happy", "sad", "surprised", "neutral"]

    camera = cv2.VideoCapture(0)
    while True:
        frame = camera.read()[1]
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        frameClone = frame.copy()
        # cv2.waitKey(0)
        i = 0
        for (x, y, w, h) in faces:
            if len(faces) > 0:
                i += 1
                # the ROI for classification via the CNN
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (48, 48))  # 48, 48
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # EMOTIONS RECOGNITION
                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]

                cv2.rectangle(frameClone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frameClone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                file_report.write(f"Face {i}--> Emotion recognized: {label}\n")


        # DEFINIZIONE DELLA SCHERMATA
        cv2.imshow('My Face - Press "Q" to exit', frameClone)
        # CHIUSURA FINESTRA
        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # RILASCIO WEBCAM
    camera.release()
    cv2.destroyAllWindows()
    file_report.write("\nEnd\n")
    file_report.write(time.strftime("Time: %H:%M:%S\n\n"))

############################################################################
#                   FINE RICONOSCIMENTO SOLO DAL VOLTO                     #
############################################################################




############################################################################
#               INIZIO RICONOSCIMENTO DA VOCE E VOLTO                      #
############################################################################

class Page_Voce_e_Volto(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.configure(background="midnight blue")
        label_benvenuto = Label(self, width=50, height=3, font=('black', 13, 'bold'), bg="green yellow",
                                text="Welcome to Emotions Recognition through Face and Voice")

        indietro = tk.Button(self, width=10, background="white", text="Back", highlightbackground="gray",
                                activebackground="gray",
                                command=lambda: Pulisci_finestra_voce_volto(self))

        blocca_registrazione = Button(self, width=10, text="Stop", bg="white", highlightbackground="gray",
                                      activebackground="gray",
                                      command=lambda: Blocca_voce(self, avvia_ric))
        avvia_ric = Button(self, width=10, text="Start", highlightbackground="gray", activebackground="gray",
                              command=lambda: voce_volto(self, avvia_ric, blocca_registrazione, results))
        results = Listbox(self, width=70, height=17, background="white")
        results.insert(END, "Press Q on the keyboard to close 'My Face' window that will open...")
        results.insert(END, "\n\n")

        label_benvenuto.grid(row=0, column=0, sticky="N", pady=10, padx=95)
        avvia_ric.grid(row=1, column=0, sticky="W", pady=10, padx=200)
        indietro.grid(row=1, column=0, sticky="E", pady=10, padx=200)
        results.grid(row=2, column=0, sticky="N", pady=30)
def Blocca_voce(self, avvia_ric):
    print("DENTRO BLOCCA VOCE.")
    flag[0] = False

    avvia_ric.grid(row=1, column=0, sticky="W", pady=10, padx=200)

    main_thread = threading.currentThread()
    tkm.showinfo("Warning", message="Recognition terminated!")
    # Elimino eventuali thread ancora in vita
    for t in threading.enumerate():
        if t is main_thread:
            continue
        print("Nome thread:", t.getName())
        t.join()
        print("Eliminato.")
    flag[0] = True
def Pulisci_finestra_voce_volto(self):
    file_report.write("End\n")
    file_report.write(time.strftime("Time: %H:%M:%S\n"))
    self.controller.show_frame("StartPage")

def voce_volto(self, avvia_ric, blocca_registrazione, results):
    avvia_ric.grid_forget()
    blocca_registrazione.grid(row=1, column=0, sticky="W", pady=10, padx=200)

    file_report.write("\nEmotions Recognition through Face & Voice Analysis\n")
    file_report.write(time.strftime("Start Time: %H:%M:%S\n\n"))

    # CARICAMENTO CLASSIFICATORI
    face_detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    emotion_classifier = load_model("emotions.hdf5", compile=False)
    EMOTIONS = ["angry", "disgusted", "scared", "happy", "sad", "surprised", "neutral"]


    def risultati():
        # ANALISI DATI VIDEO
        file_video = open("Risultati Real-Time.txt", "r+")
        # LETTURA NUMERO RIGHE ++
        file_video.seek(0, 0)
        number_lines = len(file_video.readlines())
        # print("Numero righe: ", number_lines)
        file_video.seek(0, 0)
        # print(file.readlines())


        counter_ang = 0
        counter_dis = 0
        counter_sca = 0
        counter_hap = 0
        counter_sa = 0
        counter_sur = 0
        counter_neu = 0
        emozione_video = 0

        row = [[]]
        for line in file_video.readlines():
            tmp = []
            for element in line[0:-1].split(' '):
                tmp.append(element)
            row.append(tmp)

        for i in range(1, number_lines):
            if row[i][2] == "angry":
                counter_ang += 1
            elif row[i][2] == "disgust":
                counter_dis += 1
            elif row[i][2] == "scared":
                counter_sca += 1
            elif row[i][2] == "happy":
                counter_hap += 1
            elif row[i][2] == "sad":
                counter_sa += 1
            elif row[i][2] == "surprised":
                counter_sur += 1
            elif row[i][2] == "neutral":
                counter_neu += 1

        # EMOZIONI: MASSIMO
        if counter_ang > counter_dis:
            emozione_video = "angry"
        elif counter_ang < counter_dis:
            emozione_video = "disgust"

        if counter_ang > counter_sca:
            emozione_video = "angry"
        elif counter_ang < counter_sca:
            emozione_video = "scared"

        if counter_ang > counter_hap:
            emozione_video = "angry"
        elif counter_ang < counter_hap:
            emozione_video = "happy"

        if counter_ang > counter_sa:
            emozione_video = "angry"
        elif counter_ang < counter_sa:
            emozione_video = "sad"

        if counter_ang > counter_sur:
            emozione_video = "angry"
        elif counter_ang < counter_sur:
            emozione_video = "sursprised"

        if counter_ang > counter_neu:
            emozione_video = "angry"
        elif counter_ang < counter_neu:
            emozione_video = "neutral"

        file_video.seek(0, 0)

        print("FACE")
        print(f"Recognized emotion: {emozione_video}")
        results.insert(END, "FACE")
        results.insert(END, f"Recognized emotion: {emozione_video}")
        file_video.close()

        file_report.write("\nFACE\n")
        file_report.write(f"Recognized emotion: {emozione_video}\n")

        # ANALISI DATI AUDIO
        file_audio = open("File_predizioni.txt", "r+")
        emozione_audio = ""
        row = [[]]
        for line in file_audio.readlines():
            tmp = []
            for element in line[0:-1].split(' '):
                tmp.append(element)
            row.append(tmp)

        if row[1][0] == "Disgusto":
            emozione_audio = "Disgusted"

        if row[1][0] == "Gioia":
            emozione_audio = "Happy"

        if row[1][0] == "Neutrale":
            emozione_audio = "Neutral"

        if row[1][0] == "Paura":
            emozione_audio = "Scared"

        if row[1][0] == "Rabbia":
            emozione_audio = "Angry"

        if row[1][0] == "Sorpresa":
            emozione_audio = "Surprised"

        if row[1][0] == "Tristezza":
            emozione_audio = "Sadness"


        print("\nVOICE")
        print(f"Recognized emotion: {emozione_audio}")
        print(f"Valence: {row[1][1]}")
        print(f"Arousal: {row[1][2]}")
        results.insert(END, "VOICE")
        results.insert(END, f"Recognized emotion: {emozione_audio}")
        results.insert(END, f"Valence: {row[1][1]}")
        results.insert(END, f"Arousal: {row[1][2]}")
        results.insert(END, "\n\n")
        file_audio.close()
        self.update()
        file_report.write("VOICE\n")
        file_report.write(f"Recognized emotion: {emozione_audio}\n")
        file_report.write(f"Valence: {row[1][1]}\n")
        file_report.write(f"Arousal: {row[1][2]}\n")

    # CLAUDIO
    def Esecuzione_riconoscimento_volto():

        variabile = ""
        while variabile != "inizio":

            cap = cv2.VideoCapture(0)
            file1 = open("file_avvertimento_inizio.txt", "r+")
            variabile = file1.read()
            file1.close()


            while variabile == 'inizio':
                file = open("Risultati Real-Time.txt", "w+")# qui funziona
                print("Il volto ha letto inizio!!!!!!!!")
                frame = cap.read()[1]
                frame = imutils.resize(frame, width=600)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)
                frameClone = frame.copy()
                i = 0
                #for (x, y, w, h) in faces:
                #    i += 1
                if len(faces) > 0:
                    faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                    (x, y, w, h) = faces
                    # the ROI for classification via the CNN
                    roi = gray[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (48, 48))  # 48, 48
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # EMOTIONS RECOGNITION
                    preds = emotion_classifier.predict(roi)[0]
                    emotion_probability = np.max(preds)
                    label = EMOTIONS[preds.argmax()]

                    # FILE
                    for faces in faces:
                        text = ("Face " + str(i) + ": " + label + "\n")
                        file.write(text)

                    cv2.rectangle(frameClone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frameClone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

                cv2.imshow('My Face - Press "Q" to exit', frameClone)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break



                # Apro il file per controllare se c'e scritto inizio (se c'è scritto inizio vuol dire che può partire con il registrare i frams:
                file1 = open("file_avvertimento_inizio.txt", "r+")
                variabile1 = file1.read()
                file1.close()

                # if(variabile1 == 'fine'): print("IL VOLTO HA LETTO LA FINE!!!!!")
                if (variabile1 == 'fine'):
                    # riapro il file inizio e metto la stringa vuota per far capire che il volto ha capito che deve iniziare ad acquisire i frame:
                    print("IL VOLTO HA LETTO LA FINE!")
                    #time.sleep(1)
                    ### INSERIRE FUNZIONE
                    file.close()
                    #cap.release()
                    risultati()
                    cap = cv2.VideoCapture(0)

        #cap.release()
        cv2.destroyAllWindows()

    # MICHELE
    def Esecuzione_riconoscimento_voce():

        class Thread_registra(Thread):
            def __init__(self, var):
                Thread.__init__(self)
                self.var = var

            def run(self):
                #label_emozione_riconosciuta['text'] = "Predizione in corso.."
                while (flag[0] == True):

                    time.sleep(3) #4, con 2 non funzione bene
                    file_avvertimento_inizio = open("file_avvertimento_inizio.txt", "w+")
                    file_avvertimento_inizio.write("inizio")
                    file_avvertimento_inizio.close()

                    # Il thread principale si preoccuperà di continuare sempre a registrare l'audio. Ogni audio si interrompe quando la persona
                    # smette di parlare, appena questo succede il thread principale creerà e manderà in esecuzione il thread secondario che eseguirà
                    # altre operazioni descritte sotto. Subito dopo aver avviato il thread secondario il thread principale si rimette in ascolto e così via.
                    # record(c.FILE_TEST_DIR + str(self.var) + 'test.wav')  #################################################
                    f_audio.microfono(str(self.var))  # viene chiamata questa funzione che attiva il microfono e inizia a registrare fino a quando l'utente parla.
                    # L'audio che viene registrato viene salvato sul file chiamato "audio_file_NumeroFile.wav". La parte NumeroFile è utile perchè
                    # permette di creare per ogni registrazione un file diverso in modo tale che il thread principale non salvi sempre audio differenti
                    # su uniìo stesso file audio.
                    file_avvertimento_inizio = open("file_avvertimento_inizio.txt", "w+")
                    file_avvertimento_inizio.write("fine")
                    file_avvertimento_inizio.close()


                    t_secondario = Thread_secondario(str(self.var), self.var)
                    t_secondario.start()
                    self.var = self.var + 1

        class Thread_secondario(Thread):
            # FUNZIONAMENTO THREAD SECONDARIO: ogni thread secondario chiama prima la funziona per eliminare il silenzio dal file audio che gli è stato
            # affidato, successivamente esegue delle funzioni sul file per estrarre le features necessarie, dopodichè fornisce in input al modello tali features
            # e dopo che il modello ha dato la sua risposta, il thread la stampa su un'etichetta; infine sempre il thread elimina il file audio affidatogli
            # e termina la sua esecuzione.

            def __init__(self, file_test, var):
                Thread.__init__(self)
                self.file_test = file_test
                self.var = var

            def run(self):
                if (flag[0] == True):
                    # Passo 1-eliminazione silenzio:
                    #######RIMUOVO IL SILENZIO DALL'AUDIO###################
                    nome_file_audio_del_thread_corrente = 'audio_file' + str(self.var) + '.wav'
                    # nome_file_audio_output_del_thread_corrente = 'audio_file_output' +str(self.var ) +'.wav'
                    # sound = AudioSegment.from_file('C:\\Users\\win 10\\PycharmProjects\\Programmi\\audio_file.wav',format="wav")
                    sound = AudioSegment.from_file(nome_file_audio_del_thread_corrente, format="wav")

                    start_trim = f_audio.detect_leading_silence(sound)
                    end_trim = f_audio.detect_leading_silence(sound.reverse())

                    duration = len(sound)
                    trimmed_sound = sound[start_trim:duration - end_trim]

                    # il file di output che conterrà l'audio senza silenzio è sempre quello di prima in modo da non creare troppi file ogni volta.
                    trimmed_sound.export(nome_file_audio_del_thread_corrente,format="wav")
                    durata = load_wav(nome_file_audio_del_thread_corrente, 16000, True)

                    if (durata != 0):

                        # il file di output che conterrà l'audio senza silenzio è sempre quello di prima in modo da non creare troppi file ogni volta.
                        # trimmed_sound.export('C:\\Users\\win 10\\PycharmProjects\\Programmi\\' +nome_file_audio_output_del_thread_corrente,format="wav")
                        ######################################################## ELIMINAZIONE SILENZIO TERMINATA

                        #DEVI METTERE LA FUNZIONE CHE SE L'AUDIO DURA MENO DI 1 SECONDO VIENE SCARTATO

                        # Passo 2-estrazione features: #PER L'AROUSAL (SVM)
                        # self.filename_iniziale = nome_file_audio_del_thread_corrente
                        self.filename = nome_file_audio_del_thread_corrente
                        # print("Thread numero  " +str(self.var ) +" ha il file  " +self.filename)

                        y, sr = librosa.load(self.filename, sr=16000)  # librosa estrae delle features sr = lista  delle features e y = lunghezza della lista delle features

                        # ESTRAGGO SUBITO LO SPETTROGRAMMA DAL FILE AUDIO:
                        ps = librosa.feature.melspectrogram(y=y, sr=sr)
                        # print("ps.shape prima: ", ps.shape)

                        if (ps.shape[0] != 128 or ps.shape[1] != 128):  # se lo spettrogramma non è della dimension (128,128)
                            # allora con questa piccola parte di codice lo riporto a tale dimensione con la tecnica dello
                            # zero-padding:
                            ps_temp = np.zeros([128, 128])

                            if (ps.shape[1] <= 128):
                                for i in range(0, 128):
                                    for j in range(0, ps.shape[1]):
                                        ps_temp[i][j] = ps[i][j]
                                ps = ps_temp
                            else:
                                for i in range(0, 128):
                                    for j in range(0, 128):
                                        ps_temp[i][j] = ps[i][j]
                                ps = ps_temp
                        # TECNICA ZERO-PADDING COMPLETATA (qualora fosse stata applicata)
                        # trasformo l'input per darlo alla rete neurale:
                        ps = ps.reshape(1, 128, 128, 1)
                        # print("ps.shape dopo averlo trasformato: ", ps.shape)

                        # A QUESTO PUNTO PARTE L'ESTRAZIONE DELLE FEATURES PER LA SVM:
                        y2 = librosa.effects.harmonic(y)
                        S = np.abs(librosa.stft(y))
                        mfcc_vector = librosa.feature.mfcc(y=y, sr=sr)
                        delta_vector = librosa.feature.delta(mfcc_vector)
                        delta_delta_vector = librosa.feature.delta(mfcc_vector, order=2)

                        # calcolo la frequenza fondamentale:
                        pitch, magnitudes = librosa.piptrack(y=y, sr=sr)
                        media_pitch = np.mean(pitch)
                        min_pitch = np.min(pitch)
                        max_pitch = np.max(pitch)
                        media_log_pitch = np.log(media_pitch)
                        max_log_pitch = np.log(max_pitch)
                        dev_std_pitch = np.std(pitch)
                        # estraggo features per zero crossing rate:
                        zero_c_r = librosa.feature.zero_crossing_rate(y)
                        media_zero_c_r = np.mean(zero_c_r)
                        min_z_c_r = np.min(zero_c_r)
                        max_z_c_r = np.max(zero_c_r)
                        dev_std_z_c_r = np.std(zero_c_r)

                        # estraggo le features dell'energia:
                        S, phase = librosa.magphase(librosa.stft(y))
                        rms = librosa.feature.rms(S=S)
                        media_rms = np.mean(rms)
                        min_rms = np.min(rms)
                        max_rms = np.max(rms)
                        dev_std_rms = np.std(rms)
                        rms_mediana = np.median(rms)

                        vett_features = []  # ogni volta devo creare un nuovo vettore che conterrà le features solo dell'audio corrente

                        media_c_0 = np.mean(mfcc_vector[0])
                        mediana_c_0 = np.median(mfcc_vector[0])
                        dev_stand_c_0 = np.std(mfcc_vector[0])

                        delta_media_c_0 = np.mean(delta_vector[0])
                        delta_mediana_c_0 = np.median(delta_vector[0])
                        delta_dev_stand_c_0 = np.std(delta_vector[0])

                        delta_delta_media_c_0 = np.mean(delta_delta_vector[0])
                        delta_delta_mediana_c_0 = np.median(delta_delta_vector[0])
                        delta_delta_dev_stand_c_0 = np.std(delta_delta_vector[0])

                        media_c_1 = np.mean(mfcc_vector[1])
                        mediana_c_1 = np.median(mfcc_vector[1])
                        dev_stand_c_1 = np.std(mfcc_vector[1])

                        delta_media_c_1 = np.mean(delta_vector[1])
                        delta_mediana_c_1 = np.median(delta_vector[1])
                        delta_dev_stand_c_1 = np.std(delta_vector[1])

                        delta_delta_media_c_1 = np.mean(delta_delta_vector[1])
                        delta_delta_mediana_c_1 = np.median(delta_delta_vector[1])
                        delta_delta_dev_stand_c_1 = np.std(delta_delta_vector[1])

                        media_c_2 = np.mean(mfcc_vector[2])
                        mediana_c_2 = np.median(mfcc_vector[2])
                        dev_stand_c_2 = np.std(mfcc_vector[2])

                        delta_media_c_2 = np.mean(delta_vector[2])
                        delta_mediana_c_2 = np.median(delta_vector[2])
                        delta_dev_stand_c_2 = np.std(delta_vector[2])

                        delta_delta_media_c_2 = np.mean(delta_delta_vector[2])
                        delta_delta_mediana_c_2 = np.median(delta_delta_vector[2])
                        delta_delta_dev_stand_c_2 = np.std(delta_delta_vector[2])

                        media_c_3 = np.mean(mfcc_vector[3])
                        mediana_c_3 = np.median(mfcc_vector[3])
                        dev_stand_c_3 = np.std(mfcc_vector[3])

                        delta_media_c_3 = np.mean(delta_vector[3])
                        delta_mediana_c_3 = np.median(delta_vector[3])
                        delta_dev_stand_c_3 = np.std(delta_vector[3])

                        delta_delta_media_c_3 = np.mean(delta_delta_vector[3])
                        delta_delta_mediana_c_3 = np.median(delta_delta_vector[3])
                        delta_delta_dev_stand_c_3 = np.std(delta_delta_vector[3])

                        media_c_4 = np.mean(mfcc_vector[4])
                        mediana_c_4 = np.median(mfcc_vector[4])
                        dev_stand_c_4 = np.std(mfcc_vector[4])

                        delta_media_c_4 = np.mean(delta_vector[4])
                        delta_mediana_c_4 = np.median(delta_vector[4])
                        delta_dev_stand_c_4 = np.std(delta_vector[4])

                        delta_delta_media_c_4 = np.mean(delta_delta_vector[4])
                        delta_delta_mediana_c_4 = np.median(delta_delta_vector[4])
                        delta_delta_dev_stand_c_4 = np.std(delta_delta_vector[4])

                        media_c_5 = np.mean(mfcc_vector[5])
                        mediana_c_5 = np.median(mfcc_vector[5])
                        dev_stand_c_5 = np.std(mfcc_vector[5])

                        delta_media_c_5 = np.mean(delta_vector[5])
                        delta_mediana_c_5 = np.median(delta_vector[5])
                        delta_dev_stand_c_5 = np.std(delta_vector[5])

                        delta_delta_media_c_5 = np.mean(delta_delta_vector[5])
                        delta_delta_mediana_c_5 = np.median(delta_delta_vector[5])
                        delta_delta_dev_stand_c_5 = np.std(delta_delta_vector[5])

                        media_c_6 = np.mean(mfcc_vector[6])
                        mediana_c_6 = np.median(mfcc_vector[6])
                        dev_stand_c_6 = np.std(mfcc_vector[6])

                        delta_media_c_6 = np.mean(delta_vector[6])
                        delta_mediana_c_6 = np.median(delta_vector[6])
                        delta_dev_stand_c_6 = np.std(delta_vector[6])

                        delta_delta_media_c_6 = np.mean(delta_delta_vector[6])
                        delta_delta_mediana_c_6 = np.median(delta_delta_vector[6])
                        delta_delta_dev_stand_c_6 = np.std(delta_delta_vector[6])

                        media_c_7 = np.mean(mfcc_vector[7])
                        mediana_c_7 = np.median(mfcc_vector[7])
                        dev_stand_c_7 = np.std(mfcc_vector[7])

                        delta_media_c_7 = np.mean(delta_vector[7])
                        delta_mediana_c_7 = np.median(delta_vector[7])
                        delta_dev_stand_c_7 = np.std(delta_vector[7])

                        delta_delta_media_c_7 = np.mean(delta_delta_vector[7])
                        delta_delta_mediana_c_7 = np.median(delta_delta_vector[7])
                        delta_delta_dev_stand_c_7 = np.std(delta_delta_vector[7])

                        media_c_8 = np.mean(mfcc_vector[8])
                        mediana_c_8 = np.median(mfcc_vector[8])
                        dev_stand_c_8 = np.std(mfcc_vector[8])

                        delta_media_c_8 = np.mean(delta_vector[8])
                        delta_mediana_c_8 = np.median(delta_vector[8])
                        delta_dev_stand_c_8 = np.std(delta_vector[8])

                        delta_delta_media_c_8 = np.mean(delta_delta_vector[8])
                        delta_delta_mediana_c_8 = np.median(delta_delta_vector[8])
                        delta_delta_dev_stand_c_8 = np.std(delta_delta_vector[8])

                        media_c_9 = np.mean(mfcc_vector[9])
                        mediana_c_9 = np.median(mfcc_vector[9])
                        dev_stand_c_9 = np.std(mfcc_vector[9])

                        delta_media_c_9 = np.mean(delta_vector[9])
                        delta_mediana_c_9 = np.median(delta_vector[9])
                        delta_dev_stand_c_9 = np.std(delta_vector[9])

                        delta_delta_media_c_9 = np.mean(delta_delta_vector[9])
                        delta_delta_mediana_c_9 = np.median(delta_delta_vector[9])
                        delta_delta_dev_stand_c_9 = np.std(delta_delta_vector[9])

                        media_c_10 = np.mean(mfcc_vector[10])
                        mediana_c_10 = np.median(mfcc_vector[10])
                        dev_stand_c_10 = np.std(mfcc_vector[10])

                        delta_media_c_10 = np.mean(delta_vector[10])
                        delta_mediana_c_10 = np.median(delta_vector[10])
                        delta_dev_stand_c_10 = np.std(delta_vector[10])

                        delta_delta_media_c_10 = np.mean(delta_delta_vector[10])
                        delta_delta_mediana_c_10 = np.median(delta_delta_vector[10])
                        delta_delta_dev_stand_c_10 = np.std(delta_delta_vector[10])

                        media_c_11 = np.mean(mfcc_vector[11])
                        mediana_c_11 = np.median(mfcc_vector[11])
                        dev_stand_c_11 = np.std(mfcc_vector[11])

                        delta_media_c_11 = np.mean(delta_vector[11])
                        delta_mediana_c_11 = np.median(delta_vector[11])
                        delta_dev_stand_c_11 = np.std(delta_vector[11])

                        delta_delta_media_c_11 = np.mean(delta_delta_vector[11])
                        delta_delta_mediana_c_11 = np.median(delta_delta_vector[11])
                        delta_delta_dev_stand_c_11 = np.std(delta_delta_vector[11])

                        media_c_12 = np.mean(mfcc_vector[12])
                        mediana_c_12 = np.median(mfcc_vector[12])
                        dev_stand_c_12 = np.std(mfcc_vector[12])

                        delta_media_c_12 = np.mean(delta_vector[12])
                        delta_mediana_c_12 = np.median(delta_vector[12])
                        delta_dev_stand_c_12 = np.std(delta_vector[12])

                        delta_delta_media_c_12 = np.mean(delta_delta_vector[12])
                        delta_delta_mediana_c_12 = np.median(delta_delta_vector[12])
                        delta_delta_dev_stand_c_12 = np.std(delta_delta_vector[12])

                        media_c_13 = np.mean(mfcc_vector[13])
                        mediana_c_13 = np.median(mfcc_vector[13])
                        dev_stand_c_13 = np.std(mfcc_vector[13])

                        delta_media_c_13 = np.mean(delta_vector[13])
                        delta_mediana_c_13 = np.median(delta_vector[13])
                        delta_dev_stand_c_13 = np.std(delta_vector[13])

                        delta_delta_media_c_13 = np.mean(delta_delta_vector[13])
                        delta_delta_mediana_c_13 = np.median(delta_delta_vector[13])
                        delta_delta_dev_stand_c_13 = np.std(delta_delta_vector[13])

                        media_c_14 = np.mean(mfcc_vector[14])
                        mediana_c_14 = np.median(mfcc_vector[14])
                        dev_stand_c_14 = np.std(mfcc_vector[14])

                        delta_media_c_14 = np.mean(delta_vector[14])
                        delta_mediana_c_14 = np.median(delta_vector[14])
                        delta_dev_stand_c_14 = np.std(delta_vector[14])

                        delta_delta_media_c_14 = np.mean(delta_delta_vector[14])
                        delta_delta_mediana_c_14 = np.median(delta_delta_vector[14])
                        delta_delta_dev_stand_c_14 = np.std(delta_delta_vector[14])

                        media_c_15 = np.mean(mfcc_vector[15])
                        mediana_c_15 = np.median(mfcc_vector[15])
                        dev_stand_c_15 = np.std(mfcc_vector[15])

                        delta_media_c_15 = np.mean(delta_vector[15])
                        delta_mediana_c_15 = np.median(delta_vector[15])
                        delta_dev_stand_c_15 = np.std(delta_vector[15])

                        delta_delta_media_c_15 = np.mean(delta_delta_vector[15])
                        delta_delta_mediana_c_15 = np.median(delta_delta_vector[15])
                        delta_delta_dev_stand_c_15 = np.std(delta_delta_vector[15])

                        media_c_16 = np.mean(mfcc_vector[16])
                        mediana_c_16 = np.median(mfcc_vector[16])
                        dev_stand_c_16 = np.std(mfcc_vector[16])

                        delta_media_c_16 = np.mean(delta_vector[16])
                        delta_mediana_c_16 = np.median(delta_vector[16])
                        delta_dev_stand_c_16 = np.std(delta_vector[16])

                        delta_delta_media_c_16 = np.mean(delta_delta_vector[16])
                        delta_delta_mediana_c_16 = np.median(delta_delta_vector[16])
                        delta_delta_dev_stand_c_16 = np.std(delta_delta_vector[16])

                        media_c_17 = np.mean(mfcc_vector[17])
                        mediana_c_17 = np.median(mfcc_vector[17])
                        dev_stand_c_17 = np.std(mfcc_vector[17])

                        delta_media_c_17 = np.mean(delta_vector[17])
                        delta_mediana_c_17 = np.median(delta_vector[17])
                        delta_dev_stand_c_17 = np.std(delta_vector[17])

                        delta_delta_media_c_17 = np.mean(delta_delta_vector[17])
                        delta_delta_mediana_c_17 = np.median(delta_delta_vector[17])
                        delta_delta_dev_stand_c_17 = np.std(delta_delta_vector[17])

                        media_c_18 = np.mean(mfcc_vector[18])
                        mediana_c_18 = np.median(mfcc_vector[18])
                        dev_stand_c_18 = np.std(mfcc_vector[18])

                        delta_media_c_18 = np.mean(delta_vector[18])
                        delta_mediana_c_18 = np.median(delta_vector[18])
                        delta_dev_stand_c_18 = np.std(delta_vector[18])

                        delta_delta_media_c_18 = np.mean(delta_delta_vector[18])
                        delta_delta_mediana_c_18 = np.median(delta_delta_vector[18])
                        delta_delta_dev_stand_c_18 = np.std(delta_delta_vector[18])

                        media_c_19 = np.mean(mfcc_vector[19])
                        mediana_c_19 = np.median(mfcc_vector[19])
                        dev_stand_c_19 = np.std(mfcc_vector[19])

                        delta_media_c_19 = np.mean(delta_vector[19])
                        delta_mediana_c_19 = np.median(delta_vector[19])
                        delta_dev_stand_c_19 = np.std(delta_vector[19])

                        delta_delta_media_c_19 = np.mean(delta_delta_vector[19])
                        delta_delta_mediana_c_19 = np.median(delta_delta_vector[19])
                        delta_delta_dev_stand_c_19 = np.std(delta_delta_vector[19])

                        # metto insieme tutte le features estratte in modo da creare un unico vettore da 180 elementi:
                        vett_features.append(media_c_0)
                        vett_features.append(mediana_c_0)
                        vett_features.append(dev_stand_c_0)

                        vett_features.append(delta_media_c_0)
                        vett_features.append(delta_mediana_c_0)
                        vett_features.append(delta_dev_stand_c_0)

                        vett_features.append(delta_delta_media_c_0)
                        vett_features.append(delta_delta_mediana_c_0)
                        vett_features.append(delta_delta_dev_stand_c_0)

                        vett_features.append(media_c_1)
                        vett_features.append(mediana_c_1)
                        vett_features.append(dev_stand_c_1)

                        vett_features.append(delta_media_c_1)
                        vett_features.append(delta_mediana_c_1)
                        vett_features.append(delta_dev_stand_c_1)

                        vett_features.append(delta_delta_media_c_1)
                        vett_features.append(delta_delta_mediana_c_1)
                        vett_features.append(delta_delta_dev_stand_c_1)

                        vett_features.append(media_c_2)
                        vett_features.append(mediana_c_2)
                        vett_features.append(dev_stand_c_2)

                        vett_features.append(delta_media_c_2)
                        vett_features.append(delta_mediana_c_2)
                        vett_features.append(delta_dev_stand_c_2)

                        vett_features.append(delta_delta_media_c_2)
                        vett_features.append(delta_delta_mediana_c_2)
                        vett_features.append(delta_delta_dev_stand_c_2)

                        vett_features.append(media_c_3)
                        vett_features.append(mediana_c_3)
                        vett_features.append(dev_stand_c_3)

                        vett_features.append(delta_media_c_3)
                        vett_features.append(delta_mediana_c_3)
                        vett_features.append(delta_dev_stand_c_3)

                        vett_features.append(delta_delta_media_c_3)
                        vett_features.append(delta_delta_mediana_c_3)
                        vett_features.append(delta_delta_dev_stand_c_3)

                        vett_features.append(media_c_4)
                        vett_features.append(mediana_c_4)
                        vett_features.append(dev_stand_c_4)

                        vett_features.append(delta_media_c_4)
                        vett_features.append(delta_mediana_c_4)
                        vett_features.append(delta_dev_stand_c_4)

                        vett_features.append(delta_delta_media_c_4)
                        vett_features.append(delta_delta_mediana_c_4)
                        vett_features.append(delta_delta_dev_stand_c_4)

                        vett_features.append(media_c_5)
                        vett_features.append(mediana_c_5)
                        vett_features.append(dev_stand_c_5)

                        vett_features.append(delta_media_c_5)
                        vett_features.append(delta_mediana_c_5)
                        vett_features.append(delta_dev_stand_c_5)

                        vett_features.append(delta_delta_media_c_5)
                        vett_features.append(delta_delta_mediana_c_5)
                        vett_features.append(delta_delta_dev_stand_c_5)

                        vett_features.append(media_c_6)
                        vett_features.append(mediana_c_6)
                        vett_features.append(dev_stand_c_6)

                        vett_features.append(delta_media_c_6)
                        vett_features.append(delta_mediana_c_6)
                        vett_features.append(delta_dev_stand_c_6)

                        vett_features.append(delta_delta_media_c_6)
                        vett_features.append(delta_delta_mediana_c_6)
                        vett_features.append(delta_delta_dev_stand_c_6)

                        vett_features.append(media_c_7)
                        vett_features.append(mediana_c_7)
                        vett_features.append(dev_stand_c_7)

                        vett_features.append(delta_media_c_7)
                        vett_features.append(delta_mediana_c_7)
                        vett_features.append(delta_dev_stand_c_7)

                        vett_features.append(delta_delta_media_c_7)
                        vett_features.append(delta_delta_mediana_c_7)
                        vett_features.append(delta_delta_dev_stand_c_7)

                        vett_features.append(media_c_8)
                        vett_features.append(mediana_c_8)
                        vett_features.append(dev_stand_c_8)

                        vett_features.append(delta_media_c_8)
                        vett_features.append(delta_mediana_c_8)
                        vett_features.append(delta_dev_stand_c_8)

                        vett_features.append(delta_delta_media_c_8)
                        vett_features.append(delta_delta_mediana_c_8)
                        vett_features.append(delta_delta_dev_stand_c_8)

                        vett_features.append(media_c_9)
                        vett_features.append(mediana_c_9)
                        vett_features.append(dev_stand_c_9)

                        vett_features.append(delta_media_c_9)
                        vett_features.append(delta_mediana_c_9)
                        vett_features.append(delta_dev_stand_c_9)

                        vett_features.append(delta_delta_media_c_9)
                        vett_features.append(delta_delta_mediana_c_9)
                        vett_features.append(delta_delta_dev_stand_c_9)

                        vett_features.append(media_c_10)
                        vett_features.append(mediana_c_10)
                        vett_features.append(dev_stand_c_10)

                        vett_features.append(delta_media_c_10)
                        vett_features.append(delta_mediana_c_10)
                        vett_features.append(delta_dev_stand_c_10)

                        vett_features.append(delta_delta_media_c_10)
                        vett_features.append(delta_delta_mediana_c_10)
                        vett_features.append(delta_delta_dev_stand_c_10)

                        vett_features.append(media_c_11)
                        vett_features.append(mediana_c_11)
                        vett_features.append(dev_stand_c_11)

                        vett_features.append(delta_media_c_11)
                        vett_features.append(delta_mediana_c_11)
                        vett_features.append(delta_dev_stand_c_11)

                        vett_features.append(delta_delta_media_c_11)
                        vett_features.append(delta_delta_mediana_c_11)
                        vett_features.append(delta_delta_dev_stand_c_11)

                        vett_features.append(media_c_12)
                        vett_features.append(mediana_c_12)
                        vett_features.append(dev_stand_c_12)

                        vett_features.append(delta_media_c_12)
                        vett_features.append(delta_mediana_c_12)
                        vett_features.append(delta_dev_stand_c_12)

                        vett_features.append(delta_delta_media_c_12)
                        vett_features.append(delta_delta_mediana_c_12)
                        vett_features.append(delta_delta_dev_stand_c_12)

                        vett_features.append(media_c_13)
                        vett_features.append(mediana_c_13)
                        vett_features.append(dev_stand_c_13)

                        vett_features.append(delta_media_c_13)
                        vett_features.append(delta_mediana_c_13)
                        vett_features.append(delta_dev_stand_c_13)

                        vett_features.append(delta_delta_media_c_13)
                        vett_features.append(delta_delta_mediana_c_13)
                        vett_features.append(delta_delta_dev_stand_c_13)

                        vett_features.append(media_c_14)
                        vett_features.append(mediana_c_14)
                        vett_features.append(dev_stand_c_14)

                        vett_features.append(delta_media_c_14)
                        vett_features.append(delta_mediana_c_14)
                        vett_features.append(delta_dev_stand_c_14)

                        vett_features.append(delta_delta_media_c_14)
                        vett_features.append(delta_delta_mediana_c_14)
                        vett_features.append(delta_delta_dev_stand_c_14)

                        vett_features.append(media_c_15)
                        vett_features.append(mediana_c_15)
                        vett_features.append(dev_stand_c_15)

                        vett_features.append(delta_media_c_15)
                        vett_features.append(delta_mediana_c_15)
                        vett_features.append(delta_dev_stand_c_15)

                        vett_features.append(delta_delta_media_c_15)
                        vett_features.append(delta_delta_mediana_c_15)
                        vett_features.append(delta_delta_dev_stand_c_15)

                        vett_features.append(media_c_16)
                        vett_features.append(mediana_c_16)
                        vett_features.append(dev_stand_c_16)

                        vett_features.append(delta_media_c_16)
                        vett_features.append(delta_mediana_c_16)
                        vett_features.append(delta_dev_stand_c_16)

                        vett_features.append(delta_delta_media_c_16)
                        vett_features.append(delta_delta_mediana_c_16)
                        vett_features.append(delta_delta_dev_stand_c_16)

                        vett_features.append(media_c_17)
                        vett_features.append(mediana_c_17)
                        vett_features.append(dev_stand_c_17)

                        vett_features.append(delta_media_c_17)
                        vett_features.append(delta_mediana_c_17)
                        vett_features.append(delta_dev_stand_c_17)

                        vett_features.append(delta_delta_media_c_17)
                        vett_features.append(delta_delta_mediana_c_17)
                        vett_features.append(delta_delta_dev_stand_c_17)

                        vett_features.append(media_c_18)
                        vett_features.append(mediana_c_18)
                        vett_features.append(dev_stand_c_18)

                        vett_features.append(delta_media_c_18)
                        vett_features.append(delta_mediana_c_18)
                        vett_features.append(delta_dev_stand_c_18)

                        vett_features.append(delta_delta_media_c_18)
                        vett_features.append(delta_delta_mediana_c_18)
                        vett_features.append(delta_delta_dev_stand_c_18)

                        vett_features.append(media_c_19)
                        vett_features.append(mediana_c_19)
                        vett_features.append(dev_stand_c_19)

                        vett_features.append(delta_media_c_19)
                        vett_features.append(delta_mediana_c_19)
                        vett_features.append(delta_dev_stand_c_19)

                        vett_features.append(delta_delta_media_c_19)
                        vett_features.append(delta_delta_mediana_c_19)
                        vett_features.append(delta_delta_dev_stand_c_19)

                        vett_features.append(media_pitch)
                        vett_features.append(min_pitch)
                        vett_features.append(max_pitch)
                        vett_features.append(media_log_pitch)
                        vett_features.append(max_log_pitch)
                        vett_features.append(dev_std_pitch)

                        vett_features.append(media_zero_c_r)
                        vett_features.append(min_z_c_r)
                        vett_features.append(max_z_c_r)
                        vett_features.append(dev_std_z_c_r)

                        vett_features.append(media_rms)
                        vett_features.append(min_rms)
                        vett_features.append(max_rms)
                        vett_features.append(dev_std_rms)
                        vett_features.append(rms_mediana)
                        # TERMINA L'ESTRAZIONE DELLE FEATURES

                        # Passo 3 - acquisizione oggetti per eseguire la predizione:
                        self.model = modello_caricato  # il thread secondario corrente prende l'istanza del modello della SVM per effettuare la predizione
                        self.scaler = scaler  # prende l'oggetto scaler

                        self.modello_SVM_GIOvsSOR = modello_svm_gioia_vs_sorpresa
                        self.scaler_SVM_GIOvsSOR = scaler_svm_gioia_vs_sorpresa
                        self.modello_SVM_PAUvsRAB = modello_svm_paura_vs_rabbia
                        self.scaler_SVM_PAUvsRAB = scaler_svm_paura_vs_rabbia

                        self.modello_CNN = loaded_model  # il thread secondario corrente prende l'istanza del modello della CNN per effettuare la predizione
                        ###########################################################

                        # Passo4 - fornire in input il vettore delle features al modello e ottenere la predizione:
                        vett_features_one_vs_one = []
                        vett_features_one_vs_one = vett_features

                        vett_features = np.expand_dims(vett_features, axis=0)  # cambio la dim al vettore
                        vett_features = self.scaler.transform(vett_features)  # scalo il vettore

                        # effettuo la predizione
                        # print("Pred prob - AROUSAL: ", modello_caricato.predict_proba(vett_features))
                        prediction = self.model.predict(vett_features)
                        # print("Pred audio - AROUSAL:", prediction)


                        # MEMORIZZO LA PREDIZIONE CHE MI HA DATO LA SVM SULL'AROUSAL
                        print("VALORE AROUSAL: ", prediction[0])
                        self.predizione_arousal = prediction[0]  # In questa variabile mi conservo la predizione della SVM sull'arousal

                        # ORA UTILIZZO LA CNN PER PREDIRE LA VALENCE DELL'AUDIO:
                        prob_valence = self.modello_CNN.predict_proba(ps)
                        self.predizione_valence = self.modello_CNN.predict_classes(ps)

                        # print("Pred prob - VALENCE::", prob_valence)
                        print("Pred audio - VALENCE: ", self.predizione_valence)

                        if (self.predizione_valence == 2 and self.predizione_arousal == 3):
                            print("PRED: DISGUSTO")
                            self.predizione = 'Disgusto'

                        elif (self.predizione_valence == 0 and self.predizione_arousal == 1):
                            vett_features_one_vs_one = np.expand_dims(vett_features_one_vs_one, axis=0)
                            vett_features_one_vs_one = self.scaler_SVM_GIOvsSOR.transform(
                                vett_features_one_vs_one)  # scalo il vettore
                            prediction = modello_svm_gioia_vs_sorpresa.predict(vett_features_one_vs_one)
                            if (prediction[0] == 0):
                                print("PRED: GIOIA")
                                self.predizione = "Gioia"
                            else:
                                print("PRED: SORPRESA")
                                self.predizione = "Sorpresa"

                        elif (self.predizione_valence == 1 and self.predizione_arousal == 2):
                            print("PRED: NEUTRALE")
                            self.predizione = 'Neutrale'

                        elif (self.predizione_valence == 2 and self.predizione_arousal == 1):
                            vett_features_one_vs_one = np.expand_dims(vett_features_one_vs_one, axis=0)
                            vett_features_one_vs_one = self.scaler_SVM_PAUvsRAB.transform(
                                vett_features_one_vs_one)  # scalo il vettore
                            prediction = modello_svm_paura_vs_rabbia.predict(vett_features_one_vs_one)
                            if (prediction[0] == 0):
                                print("PRED: PAURA")
                                self.predizione = "Paura"
                            else:
                                print("PRED: RABBIA")
                                self.predizione = "Rabbia"

                        elif (self.predizione_valence == 2 and self.predizione_arousal == 2):
                            print("PRED: TRISTEZZA")
                            self.predizione = 'Tristezza'

                        else:
                            print("PRED: NESSUNA COMBINAZIONE")
                            self.predizione = 'No emotion'

                        if (flag[0] == True):
                            #label_emozione_riconosciuta['text'] = "Predizione-" + str(self.var) + ": " + self.predizione
                            file = open("File_predizioni.txt", "w+")

                            #valence = ""
                            #arousal = ""
                            if(self.predizione_valence == 0):
                                valence = "Positive"
                            elif(self.predizione_valence == 1):
                                valence = "Neutral"
                            else:
                                valence = "Negative"

                            if(self.predizione_arousal == 1):
                                arousal = "High"
                            elif(self.predizione_arousal == 2):
                                arousal = "Medium"
                            else:
                                arousal = "Low"

                            file.write(self.predizione + ' ' + valence + ' ' + arousal + ' ')




                        # Passo 5 - Cancellare il file audio di test del thread secondario corrente:
                        # os.remove(self.filename_iniziale)
                        # os.remove(self.filename)
                        os.unlink(self.filename)
                    else:
                        print("File troppo corto <1 secondo...è stato eliminato!")
        var0 = 1

        t_registra = Thread_registra(var0)
        t_registra.start()

    p1 = Process(target=Esecuzione_riconoscimento_voce())
    p2 = Process(target=Esecuzione_riconoscimento_volto())

    p1.start()
    p2.start()

############################################################################
#               FINE RICONOSCIMENTO DA VOCE E VOLTO                        #
############################################################################

if __name__ == "__main__":

    # deserializzo il modello SVM con joblib:
    ####################################
    modello_caricato = load('MODELLO FINALE SVM AROUSAL/SVM_modello_RAV,TESS,EMODB,SAVEE,EMOFILM e audio del bilanciamento (SOR 1).joblib')
    scaler = load('MODELLO FINALE SVM AROUSAL/SVM_scaler_RAV,TESS,EMODB,SAVEE,EMOFILM e audio del bilanciamento (SOR 1).joblib')
    print("MODELLO SVM CARICATO.")

    # carico le SVM OnevsOne
    modello_svm_paura_vs_rabbia = load('Modelli FINALI SVM ONEvsONE/SVM_modello_RAV,TESS,EMODB,SAVEE,EMOFILM e audio del bilanciamento (PAUvsRAB).joblib')
    scaler_svm_paura_vs_rabbia = load('Modelli FINALI SVM ONEvsONE/SVM_scaler_RAV,TESS,EMODB,SAVEE,EMOFILM e audio del bilanciamento (PAUvsRAB).joblib')

    modello_svm_gioia_vs_sorpresa = load('Modelli FINALI SVM ONEvsONE/SVM_modello_RAV,TESS,EMODB,SAVEE,EMOFILM e audio del bilanciamento (GIOvsSOR).joblib')
    scaler_svm_gioia_vs_sorpresa = load('Modelli FINALI SVM ONEvsONE/SVM_scaler_RAV,TESS,EMODB,SAVEE,EMOFILM e audio del bilanciamento (GIOvsSOR).joblib')

    # carico il modello CNN PER RICONOSCERE LA VALENCE:
    json_file = open('model-indiano-SOLO VALENCE(128,128)-(normale,GN-0-02)-60 epoche-paura 2 e neutro 1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model-indiano-SOLO VALENCE(128,128)-(normale,GN-0-02)-60 epoche-paura 2 e neutro 1.h5")
    # loaded_model.load_weights("model-indiano-SOLO VALENCE(128,128)-NEUTRO UGUALE A GIOIA E SORPRESA-(normale,GN-0-02)-60 epoche.h5")
    loaded_model._make_predict_function()
    print("MODELLO CNN CARICATO.")

    file_report = open("Report.txt", "w")
    file_report.write("START PROGRAM\n")
    file_report.write(time.strftime("Date: %d/%m/%Y\n"))
    file_report.write(time.strftime("Time: %H:%M:%S\n"))

    ####################################

    print("Modello caricato correttamente.")

    flag = [True]  # permette di capire quando viene premuto il pulsante per bloccare la verifica o l'identificazione
    # se flag[0] == True allora vuol dire che l'utente non ha bloccato la registrazione.
    app = SampleApp()
    app.geometry("700x500")
    app.resizable(False, False)

    app.mainloop()