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
    card_images = {}
    player_hand = []

    def __init__(self):
        self.new_game_btn = None
        self.human_agent = None
        self.menu_frame = None
        self.root = Tk()
        self.root.minsize(900, 700)
        content_padding = (50, 50, 50, 50)  # w-n-e-s
        style = ttk.Style()
        style.configure("Green.TFrame", background="green")
        style.configure("Green.TButton", foreground="green", background="green", padding=-1)
        style.configure("Black.TFrame", background="black")
        style.configure("Black.TButton", foreground="black", background="black", padding=-1)
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
        image_size = (98, 162)
        for filename in os.listdir("card_images"):
            img_path = os.path.join("card_images", filename)
            img = Image.open(img_path).resize(image_size, resample=Resampling.LANCZOS)
            self.card_images[filename] = (ImageTk.PhotoImage(image=img))

    @classmethod
    def load_images(cls):
        """
        @return: list of all card images
        """
        card_images = {}
        # loading and resizing all images
        image_size = (98, 162)
        for filename in os.listdir("card_images"):
            img_path = os.path.join("card_images", filename)
            img = Image.open(img_path).resize(image_size, resample=Resampling.LANCZOS)
            card_images[filename] = (ImageTk.PhotoImage(image=img))
        return card_images

    @classmethod
    def find_card_name(cls, card, card_images):
        """
        Finds the filename of the card image

        @param card: card object of the Briscola game
        @param card_images: list of all cards images objects

        @return: filename or None if the card doesn't exist
        """
        card_name_split = card.name.split()
        for filename in card_images:
            # e.g. briscola = ["Asso", "Di", "Bastoni"]
            if card_name_split[0] in filename and card_name_split[2].lower() in filename:
                return filename
        return None

    def set_briscola(self, briscola_name):
        """
        Set the filename of the Briscola image

        @param briscola_name the image filename
        """
        self.briscola_name = briscola_name

    def player_play_card(self, index):
        """
        This function moves the chosen card from the "player_frame" to the "table_frame".
        @param index: index of the played card (can be [0, 2])
        :return image filename of the played card
        """
        print("played card index: ", index)
        # creating and showing a label that represents the player played card into the table frame
        try:
            self.table_frame.winfo_children()[1].winfo_children()[0].destroy()
        except IndexError:
            pass
        new_card_btn = ttk.Label(self.table_frame.winfo_children()[1], style="Green.TButton",
                                 image=self.card_images[self.player_hand[index]])
        new_card_btn.grid(column=0, row=0)

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

    def agent_play_card(self, card_name):
        """
        This function moves a card from the "agent_frame" to the "table_frame".
        @param card_name: image filename of the card that the agent played
        """
        # remove a "retro card" from the agent frame
        self.agent_frame.winfo_children()[0].destroy()
        # showing the played card inside the table frame
        card = ttk.Label(self.table_frame.winfo_children()[0], style="Green.TButton", image=self.card_images[card_name])
        card.grid(column=0, row=0)

    def empty_deck(self):
        """
        This function loads the image "Deck_Finito.jpg" into the frame "deck_frame".
        """
        self.deck_frame.winfo_children()[1].destroy()
        empty_deck_label = ttk.Label(self.deck_frame, style="Green.TButton", image=self.card_images["Deck_Finito.jpg"])
        empty_deck_label.grid(column=1, row=0, padx=self.card_label_padding, pady=self.card_label_padding)

    def empty_briscola(self):
        """
        This function removes the image of the Briscola from the frame "deck_frame".
        """
        self.deck_frame.winfo_children()[0].destroy()

    def agent_draw_card(self):
        """
        This function adds a card to the agent hand.
        """
        agent_frame_children = self.agent_frame.winfo_children()
        card = ttk.Label(self.agent_frame, style="Green.TButton", image=self.card_images["Carte_Napoletane_retro.jpg"])

        card.grid(column=len(agent_frame_children), row=0, padx=self.card_label_padding,
                  pady=self.card_label_padding)

    def player_draw_card(self, card_name):
        """
        This function adds a card to the player hand at the right most place
        @param card_name: image filename of the card to be added
        """
        # creating the new card object and passing to its command function the "right" most index
        index = len(self.player_hand)
        partial_func = partial(self.player_play_card, index)
        draw_card = ttk.Button(self.player_frame, style="Green.TButton", image=self.card_images[card_name],
                               command=partial_func)

        # adding the card at the right most place in the player hand
        draw_card.grid(column=len(self.player_hand), row=0, padx=self.card_label_padding,
                       pady=self.card_label_padding)
        self.player_hand.append(card_name)

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
        game = brisc.BriscolaGame(gui_obj, 2, logger)
        game.reset(self.card_images, gui_obj)

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
                break
        self.populate_agent_frame()
        thread = threading.Thread(target=brisc.play_episode, args=(game, agents, self.cond, gui_obj, False,))
        thread.start()

    def populate_menu_frame(self, gui_obj):
        """
        Inserts the "new_game" button into "menu_frame"

        @param gui_obj:
        """
        partial_func = partial(self.start_game, gui_obj)
        self.new_game_btn = ttk.Button(self.menu_frame, text="New game", command=partial_func)
        self.new_game_btn.grid(column=0, row=0, sticky="NSEW")

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
        briscola_label = ttk.Label(self.deck_frame, style="Green.TButton", image=self.card_images[briscola_name])
        deck_label = ttk.Label(self.deck_frame, style="Green.TButton",
                               image=self.card_images["Carte_Napoletane_retro.jpg"])

        briscola_label.grid(column=0, row=0, sticky="NS", padx=self.card_label_padding, pady=self.card_label_padding)
        deck_label.grid(column=1, row=0, sticky="NS", padx=self.card_label_padding, pady=self.card_label_padding)

    def populate_agent_frame(self):
        """
        Inserts 3 hidden cards into "deck_frame".
        """
        # here we are inside the frame "agent_frame"
        card_1 = ttk.Label(self.agent_frame, style="Green.TButton",
                           image=self.card_images["Carte_Napoletane_retro.jpg"])
        card_2 = ttk.Label(self.agent_frame, style="Green.TButton",
                           image=self.card_images["Carte_Napoletane_retro.jpg"])
        card_3 = ttk.Label(self.agent_frame, style="Green.TButton",
                           image=self.card_images["Carte_Napoletane_retro.jpg"])

        card_1.grid(column=0, row=0, padx=self.card_label_padding, pady=self.card_label_padding)
        card_2.grid(column=1, row=0, padx=self.card_label_padding, pady=self.card_label_padding)
        card_3.grid(column=2, row=0, padx=self.card_label_padding, pady=self.card_label_padding)

    def create_main_frames(self):
        """
        Creates the 4 main frames.
        """
        self.menu_frame = ttk.Frame(self.content, width=100, style="Green.TFrame")
        self.agent_frame = ttk.Frame(self.content, width=350, style="Green.TFrame")
        self.player_frame = ttk.Frame(self.content, width=350, style="Green.TFrame")
        self.table_frame = ttk.Frame(self.content, width=350, style="Green.TFrame")
        self.deck_frame = ttk.Frame(self.content, width=200, style="Green.TFrame")

        self.menu_frame.grid(column=0, row=0, rowspan=3, sticky="NSW")
        self.player_frame.grid(column=1, row=2, sticky="NS")
        self.agent_frame.grid(column=1, row=0, sticky="NS")
        self.table_frame.grid(column=1, row=1, sticky="NS")
        self.deck_frame.grid(column=2, row=1, sticky="NS")

        # --- inserting nested frames ---
        # here we are inside the frame "table_frame"
        agent_played_card_frame = ttk.Frame(self.table_frame, style="Green.TFrame")
        player_played_card_frame = ttk.Frame(self.table_frame, style="Green.TFrame")
        agent_played_card_frame.grid(column=0, row=0, sticky="NS", padx=self.frame_padding, pady=self.frame_padding)
        player_played_card_frame.grid(column=1, row=0, sticky="NS", padx=self.frame_padding, pady=self.frame_padding)

        # resizing frames with resolution changes
        self.deck_frame.columnconfigure(0, weight=1)
        self.deck_frame.columnconfigure(1, weight=1)
        self.deck_frame.rowconfigure(0, weight=0)
        self.player_frame.columnconfigure(0, weight=1)
        self.player_frame.columnconfigure(1, weight=1)
        self.player_frame.columnconfigure(2, weight=1)
        self.player_frame.rowconfigure(0, weight=0)
        self.agent_frame.columnconfigure(0, weight=1)
        self.agent_frame.columnconfigure(1, weight=1)
        self.agent_frame.columnconfigure(2, weight=1)
        self.agent_frame.rowconfigure(0, weight=0)

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
