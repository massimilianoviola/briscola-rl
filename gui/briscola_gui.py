import argparse
import os
import sys
import threading
import time
from functools import partial
from tkinter import *
from tkinter import ttk

from PIL import ImageTk, Image
from PIL.Image import Resampling

import environment as brisc
from agents.ai_agent import AIAgent
from agents.human_agent import HumanAgent
from agents.q_agent import QAgent
from utils import NetworkTypes, BriscolaLogger


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


played_card = None


class BriscolaGui:
    briscola_name = None
    content = None
    root = None
    agent_frame = None
    player_frame = None
    table_frame = None
    deck_frame = None
    frame_padding = 5
    card_label_padding = 5
    cond = threading.Condition()
    # {"12_Due_di_coppe.jpg": image_object, ...}
    card_images = {}
    # lists containing filename of card images
    player_hand = []
    agent_hand = []
    image_size = (98, 162)

    def __init__(self):
        self.briscola_img = None
        self.fixed_briscola_frame = None
        self.fixed_briscola = None
        self.deck_count = None
        self.agent_score = None
        self.player_score = None
        self.agent_counter = None
        self.player_counter = None
        self.log_text = None
        self.count_log_row = "1"
        self.log_frame = None
        self.new_game_btn = None
        self.human_agent = None
        self.menu_frame = None
        self.root = Tk()
        self.root.minsize(900, 700)
        content_padding = (50, 50, 50, 50)  # w-n-e-s
        style = ttk.Style()
        style.configure("Green.TFrame", background="green")
        style.configure("Green.TButton", foreground="white", background="green", padding=-1)
        style.configure("Retro.TLabel", foreground="white", padding=-1)
        self.content = ttk.Frame(self.root, padding=content_padding, style="Green.TFrame")
        self.content.grid(column=0, row=0, sticky="NSEW")
        # root and content resizing when resolution changes
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=1)
        self.content.rowconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=1)
        self.content.rowconfigure(2, weight=1)
        # loading and resizing all images

        names = ['01_Asso_di_denari.jpg', '02_Due_di_denari.jpg', '03_Tre_di_denari.jpg', '04_Quattro_di_denari.jpg',
                 '05_Cinque_di_denari.jpg', '06_Sei_di_denari.jpg', '07_Sette_di_denari.jpg', '08_Fante_di_denari.jpg',
                 '09_Cavallo_di_denari.jpg', '10_Re_di_denari.jpg', '11_Asso_di_coppe.jpg', '12_Due_di_coppe.jpg',
                 '13_Tre_di_coppe.jpg', '14_Quattro_di_coppe.jpg', '15_Cinque_di_coppe.jpg', '16_Sei_di_coppe.jpg',
                 '17_Sette_di_coppe.jpg', '18_Fante_di_coppe.jpg', '19_Cavallo_di_coppe.jpg', '20_Re_di_coppe.jpg',
                 '21_Asso_di_spade.jpg', '22_Due_di_spade.jpg', '23_Tre_di_spade.jpg', '24_Quattro_di_spade.jpg',
                 '25_Cinque_di_spade.jpg', '26_Sei_di_spade.jpg', '27_Sette_di_spade.jpg', '28_Fante_di_spade.jpg',
                 '29_Cavallo_di_spade.jpg', '30_Re_di_spade.jpg', '31_Asso_di_bastoni.jpg', '32_Due_di_bastoni.jpg',
                 '33_Tre_di_bastoni.jpg', '34_Quattro_di_bastoni.jpg', '35_Cinque_di_bastoni.jpg',
                 '36_Sei_di_bastoni.jpg', '37_Sette_di_bastoni.jpg', '38_Fante_di_bastoni.jpg',
                 '39_Cavallo_di_bastoni.jpg', '40_Re_di_bastoni.jpg', 'Carte_Napoletane_retro.jpg', 'Deck_Finito.jpg']

        for filename in names:
            img_path = resource_path("card_images/" + filename)
            img = Image.open(img_path).resize(self.image_size, resample=Resampling.LANCZOS)
            self.card_images[filename] = (ImageTk.PhotoImage(image=img))

    @classmethod
    def find_card_name(cls, card, card_images):
        """
        Finds the filename of the card image

        @param card: card object of the Briscola game
        @param card_images: list of all cards images objects

        @return: filename if the agent effectively draws a card,
        None if the card doesn't exist or no more cards can be to draw
        """
        try:
            card_name_split = card.name.split()
            for filename in card_images:
                # e.g. briscola = ["Asso", "Di", "Bastoni"]
                if card_name_split[0] in filename and card_name_split[2].lower() in filename:
                    return filename
        except AttributeError:
            pass
        return None

    def set_briscola(self, briscola_name):
        """
        Sets the filename of the Briscola image and populates "fixed_briscola_frame"

        @param briscola_name the filename of the briscola image
        """
        self.briscola_name = briscola_name
        img_path = resource_path("card_images/" + self.briscola_name)
        new_width = int(self.image_size[0] / 1.2)
        new_height = int(self.image_size[1] / 1.2)
        img = Image.open(img_path).resize((new_width, new_height), resample=Resampling.LANCZOS)
        self.briscola_img = (ImageTk.PhotoImage(image=img))
        self.fixed_briscola_frame = ttk.Frame(self.content, style="Green.TFrame", height=300)
        self.fixed_briscola_frame.grid(column=3, row=2)
        self.fixed_briscola = ttk.Label(self.fixed_briscola_frame, style="Green.TButton",
                                        image=self.briscola_img)
        self.fixed_briscola.grid(column=0, row=1)
        briscola_text = ttk.Label(self.fixed_briscola_frame, text="Briscola", style="Counter.TLabel")
        briscola_text.grid(column=0, row=0)

    def player_play_card(self, index):
        """
        This function moves the chosen card from the "player_frame" to the "table_frame".

        @param index: index of the played card (can be [0, 2])
        """
        # creating and showing a label that represents the player played card into the table frame
        try:
            self.table_frame.winfo_children()[1].winfo_children()[0].destroy()
        except IndexError:
            pass
        new_card = ttk.Label(self.table_frame.winfo_children()[1], style="Green.TButton",
                             image=self.card_images[self.player_hand[index]])
        new_card.grid(column=0, row=0)

        # destroying the card objects from the player frame and repopulating it
        self.player_hand.pop(index)
        new_player_hand = []
        for card in self.player_hand:
            new_player_hand.append(card)
        self.player_hand.clear()
        for child in self.player_frame.winfo_children():
            child.destroy()
        self.populate_player_frame(new_player_hand)

        # notify the game that a card has being played
        with self.cond:
            self.human_agent.played_card(index)
            self.cond.notify()

    def agent_play_card(self, index):
        """
        This function moves a card from the "agent_frame" to the "table_frame".

        @param index: index of the played card (can be [0, 2])
        """
        # time.sleep(1)
        try:
            self.table_frame.winfo_children()[0].winfo_children()[0].destroy()
        except IndexError:
            pass
        # creating and showing a label that represents the agent played card into the table frame
        card = ttk.Label(self.table_frame.winfo_children()[0], style="Green.TButton",
                         image=self.card_images[self.agent_hand[index]])
        card.grid(column=0, row=0)
        # updating the agent hand
        self.agent_hand.pop(index)
        # removing one card from the agent frame
        for child in self.agent_frame.winfo_children():
            child.destroy()
        self.populate_agent_frame()

    def empty_deck_frame(self):
        """
        This function loads the image "Deck_Finito.jpg" into the frame "deck_frame".
        """
        for child in self.deck_frame.winfo_children():
            child.destroy()

    def agent_draw_card(self, card_name):
        """
        This function adds a card to the agent hand.
        """
        if card_name is None:
            return
        card = ttk.Label(self.agent_frame, style="Retro.TLabel", image=self.card_images["Carte_Napoletane_retro.jpg"])

        card.grid(column=len(self.agent_hand), row=0, padx=self.card_label_padding,
                  pady=self.card_label_padding)
        self.agent_hand.append(card_name)
        if card_name == self.briscola_name:
            self.empty_deck_frame()

    def player_draw_card(self, card_name):
        """
        This function adds a card to the player hand at the right most place
        @param card_name: image filename of the card to be added
        """
        if card_name is None:
            return
        # creating the new card object and passing to its command function the "right" most index
        index = len(self.player_hand)
        partial_func = partial(self.player_play_card, index)
        draw_card = ttk.Button(self.player_frame, style="Green.TButton", image=self.card_images[card_name],
                               command=partial_func)

        # adding the card at the right most place in the player hand
        draw_card.grid(column=len(self.player_hand), row=0, padx=self.card_label_padding,
                       pady=self.card_label_padding)
        self.player_hand.append(card_name)
        if card_name == self.briscola_name:
            self.empty_deck_frame()

    def start_game(self, gui_obj):
        """Play against one of the intelligent agents."""
        self.new_game_btn.destroy()
        parser = argparse.ArgumentParser()

        parser.add_argument("--model_dir", default=None,
                            help="Provide a trained model path if you want to play against a deep agent", type=str)
        parser.add_argument("--network", default=NetworkTypes.DRQN, choices=[NetworkTypes.DQN, NetworkTypes.DRQN],
                            help="Neural network used for approximating value function")

        flags = parser.parse_args()

        # initialize the environment
        logger = BriscolaLogger(BriscolaLogger.LoggerLevels.PVP)
        game = brisc.BriscolaGame(2, logger, gui_obj)
        game.reset()

        # initialize the agents
        self.human_agent = HumanAgent()
        agents = [self.human_agent]

        if flags.model_dir:
            agent = QAgent(network=flags.network)
            agent.load_model(flags.model_dir)
            agent.make_greedy()
            agents.append(agent)
        else:
            agent = AIAgent()
            agents.append(agent)

        # initializing the gui and starting the game
        briscola = game.briscola.name.split()
        # finding the image file of the briscola
        for filename in self.card_images:
            # e.g. briscola = ["Asso", "Di", "Bastoni"]
            if briscola[0] in filename and briscola[2].lower() in filename:
                self.populate_deck_frame(filename)
                self.set_briscola(filename)
                break

        self.update_player_score(0)
        self.update_agent_score(0)
        self.update_deck_count(33)
        self.insert_log("Game started...")

        thread = threading.Thread(target=brisc.play_episode, args=(game, agents, gui_obj, False))
        thread.start()

    def populate_menu_frame(self, gui_obj):
        """
        Inserts the "new_game" button into "menu_frame"

        @param gui_obj:
        """
        partial_func = partial(self.start_game, gui_obj)
        self.new_game_btn = ttk.Button(self.menu_frame, text="New game", command=partial_func)
        self.new_game_btn.grid(column=0, row=0, sticky="NS")

    def populate_player_frame(self, cards_name):
        """
        Inserts cards into the frame "player_frame".
        @param cards_name: array containing image filenames
        """
        for i in range(0, len(cards_name)):
            partial_func = partial(self.player_play_card, i)
            card = ttk.Button(self.player_frame, style="Green.TButton", image=self.card_images[cards_name[i]],
                              command=partial_func)
            self.player_hand.append(cards_name[i])
            card.grid(column=i, row=0, padx=self.card_label_padding, pady=self.card_label_padding)

    def populate_deck_frame(self, briscola_name):
        """
        Inserts images of the deck and of the briscola card into the frame "deck_frame".
        """
        empty_label = ttk.Label(self.deck_frame, background="green", width=15)
        briscola_label = ttk.Label(self.deck_frame, style="Green.TButton", image=self.card_images[briscola_name])
        briscola_label.place(relx=0.3, rely=0.4, anchor="center")
        deck_label = ttk.Label(self.deck_frame, style="Retro.TLabel",
                               image=self.card_images["Carte_Napoletane_retro.jpg"])

        # briscola_label.grid(column=0, row=0, sticky="NS", padx=self.card_label_padding, pady=self.card_label_padding)
        empty_label.grid(column=0, row=0)
        deck_label.grid(column=1, row=0, sticky="NS", padx=self.card_label_padding, pady=self.card_label_padding+10)

    def populate_agent_frame(self):
        """
        Inserts 3 hidden cards into "agent_frame".
        """
        for i in range(0, len(self.agent_hand)):
            card = ttk.Label(self.agent_frame, style="Retro.TLabel",
                             image=self.card_images["Carte_Napoletane_retro.jpg"])
            card.grid(column=i, row=0, padx=self.card_label_padding, pady=self.card_label_padding)

    def empty_table_frame(self):
        """
        Removes all objects from the table frame
        """
        # time.sleep(1)
        try:
            self.table_frame.winfo_children()[0].winfo_children()[0].destroy()
        except IndexError:
            pass
        try:
            self.table_frame.winfo_children()[1].winfo_children()[0].destroy()
        except IndexError:
            pass

    def insert_log(self, text):
        """
        Inserts the given text into a new line in the log frame.
        """
        self.log_text.config(state="normal")
        self.log_text.insert(self.count_log_row + ".0", "- " + text + "\n")
        self.count_log_row = str(int(self.count_log_row) + 1)
        # focussing always the last line (with the scroll bar)
        self.log_text.see("end")
        # disabling the text widget
        self.log_text.config(state="disabled")

    def update_player_score(self, score):
        """
        Updates the score counter of the player

        @param score: int that represents the new score of the player
        """
        self.player_score["text"] = str(score)

    def update_agent_score(self, score):
        """
        Updates the score counter of the agent

        @param score: int that represents the new score of the agent
        """
        self.agent_score["text"] = str(score)

    def update_deck_count(self, count):
        """
        Updates the number of cards inside the deck

        @param count: int that represents the new score of the agent
        """
        self.deck_count["text"] = str(count)

    def create_main_frames(self):
        """
        Creates the 4 main frames.
        """
        self.menu_frame = ttk.Frame(self.content, style="Green.TFrame")
        self.agent_frame = ttk.Frame(self.content, style="Green.TFrame")
        self.player_frame = ttk.Frame(self.content, style="Green.TFrame")
        self.table_frame = ttk.Frame(self.content, style="Green.TFrame")
        self.deck_frame = ttk.Frame(self.content, style="Green.TFrame")
        self.log_frame = ttk.Frame(self.content, style="Green.TFrame")

        self.menu_frame.grid(column=0, row=0, rowspan=3, sticky="W")
        self.player_frame.grid(column=1, row=2)
        self.agent_frame.grid(column=1, row=0)
        self.table_frame.grid(column=1, row=1)
        self.deck_frame.grid(column=3, row=1)
        self.log_frame.grid(column=3, row=0)

        # inserting nested frames
        agent_played_card_frame = ttk.Frame(self.table_frame, style="Green.TFrame")
        player_played_card_frame = ttk.Frame(self.table_frame, style="Green.TFrame")
        agent_played_card_frame.grid(column=0, row=0, sticky="NS", padx=self.frame_padding, pady=self.frame_padding)
        player_played_card_frame.grid(column=1, row=0, sticky="NS", padx=self.frame_padding, pady=self.frame_padding)

        self.log_text = Text(self.log_frame, background="green", foreground="white", width=35, height=10,
                             padx=self.frame_padding, pady=self.frame_padding, state="disabled")
        self.log_text.grid(column=0, row=0, sticky="EW")
        ys = ttk.Scrollbar(self.log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text['yscrollcommand'] = ys.set
        ys.grid(column=1, row=0, sticky="NS")

        # counters
        style = ttk.Style()
        style.configure("Counter.TLabel", font=("Arial", 15), background="green", foreground="white")
        self.player_score = ttk.Label(self.content, style="Counter.TLabel")
        self.player_score.grid(column=2, row=2, sticky="W", padx=30)
        self.agent_score = ttk.Label(self.content, style="Counter.TLabel")
        self.agent_score.grid(column=2, row=0, sticky="W", padx=30)
        self.deck_count = ttk.Label(self.deck_frame, style="Counter.TLabel")
        self.deck_count.grid(column=1, row=1)

        self.log_frame.columnconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=0)
        self.content.columnconfigure(2, weight=0)
        self.content.columnconfigure(3, weight=1)
        self.log_frame.columnconfigure(0, weight=1)

    def start_gui(self, gui_obj):
        """
        This method populates the content with initial values and starts the gui for the Briscola game.

        @param gui_obj:
        """
        self.create_main_frames()

        self.populate_menu_frame(gui_obj)

        self.root.mainloop()


if __name__ == '__main__':
    briscola_gui = BriscolaGui()
    briscola_gui.start_gui(briscola_gui)
