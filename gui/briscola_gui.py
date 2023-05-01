"""
@author: Alberto Ursino: albertoursino98@gmail.com
"""

import argparse
import os
import sys
import threading
from functools import partial
from tkinter import *
from tkinter import ttk, messagebox

import _tkinter
from PIL import ImageTk, Image
from PIL.Image import Resampling

import environment as brisc
from agents.ai_agent import AIAgent
from agents.human_agent import HumanAgent
from agents.q_agent import QAgent
from utils import NetworkTypes, BriscolaLogger


def resource_path(relative_path):
    """
    This function is necessary for gathering the card images paths when the executable is used.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class BriscolaGui:
    frame_padding = 5
    card_label_padding = 5
    cond = threading.Condition()
    reset = False
    # {"12_Due_di_coppe.jpg": image_object, ...}
    card_images = {}
    image_size = (98, 162)
    saved_commands = []
    count_log_row = "1"

    def __init__(self):
        self.deck_frame = None
        self.table_frame = None
        self.agent_frame = None
        self.player_frame = None
        self.menu_frame = None
        self.log_frame = None
        self.agent_score_frame = None
        self.player_score_frame = None

        self.logger = None
        self.agent = None
        self.human_agent = None
        self.game_thread = None
        self.briscola_game = None

        self.new_game_btn = None
        self.restart_btn = None
        self.deck_count_label = None

        self.briscola_img = None
        self.briscola_name = None

        self.agent_score = None
        self.player_score = None
        self.log_text = None

        self.player_hand = []
        self.agent_hand = []
        self.player_played_card = None
        self.agent_played_card = None

        # defining root and content
        self.root = Tk()
        self.root.minsize(1100, 700)
        self.root.maxsize(1100, 700)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.content = ttk.Frame(self.root, padding=(50, 50, 50, 50), style="Green.TFrame")
        self.content.grid(column=0, row=0, sticky="NSEW")

        # defining some styles
        style = ttk.Style()
        style.configure("Green.TFrame", background="green")
        style.configure("Card.TButton", background="green")
        style.configure("Retro.TLabel", background="grey")
        style.configure("NewGame.TButton", font=("Arial", 15), background='green')
        style.configure("Restart.TButton", font=("Arial", 10), background='green')
        style.configure("CounterText.TLabel", foreground="white", background="green", font=("Arial", 12))
        style.configure("WinnerText.TLabel", foreground="white", background="green", font=("Arial", 18))

        # gathering card images
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

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
            self.reset = True
            with self.cond:
                self.cond.notify()

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
        fixed_briscola_frame = ttk.Frame(self.content, style="Green.TFrame", height=300)
        fixed_briscola_frame.grid(column=3, row=2)
        fixed_briscola = ttk.Label(fixed_briscola_frame, style="Card.TButton",
                                   image=self.briscola_img)
        fixed_briscola.grid(column=0, row=1)
        briscola_text = ttk.Label(fixed_briscola_frame, text="Briscola", style="Counter.TLabel")
        briscola_text.grid(column=0, row=0, pady=1)

    def hand_to_table(self, player_type, height):
        """
        Moves smoothly a card from the hand of the agent or the player (depending on the parameter) to the table.

        @param player_type: (str) can be "human" or "agent".
        @param height: (int) starting height of the card wrt to the content
        """
        if player_type != "human" and player_type != "agent":
            raise ValueError
        if player_type == "human":
            if height > 300:
                self.player_played_card.place(x=375, y=height, anchor="center")
                height -= 3
                self.root.after(1, lambda: self.hand_to_table(player_type, height))
        if player_type == "agent":
            if height <= 300:
                self.agent_played_card.place(x=250, y=height, anchor="center")
                height += 3
                self.root.after(1, lambda: self.hand_to_table(player_type, height))

    def player_play_card(self, index):
        """
        Moves the chosen card from "player_frame" to "table_frame".

        @param index: (int) index of the played card.
            If the player has 3 cards in his hand, index can be [0, 2] where 0 is the left-most card.
        """
        # creating a new card object to be put into the table
        self.player_played_card = ttk.Label(self.content, style="Card.TButton",
                                            image=self.card_images[self.player_hand[index]])

        # removing the played card from the player hand and updating the commands with the new indexes
        self.player_hand.pop(index)
        self.player_frame.winfo_children()[index].destroy()
        for i in range(index, len(self.player_frame.winfo_children())):
            partial_func = partial(self.player_play_card, i)
            self.player_frame.winfo_children()[i]["command"] = partial_func
            self.player_frame.winfo_children()[i].grid(column=i, row=0)

        # moving smoothly the card from the hand to the table
        height = 400
        self.player_played_card.place(x=375, y=height, anchor="center")
        self.hand_to_table("human", height)

        # notify the game that a card has being played
        with self.cond:
            self.human_agent.played_card(index)
            self.cond.notify()

    def agent_play_card(self, index):
        """
        This function moves a card from the "agent_frame" to the "table_frame".

        @param index: index of the played card (can be [0, 2])
        """
        # creating a new card object to be put into the table
        self.agent_played_card = ttk.Label(self.content, style="Card.TButton",
                                           image=self.card_images[self.agent_hand[index]])

        # removing the played card from the agent hand
        self.agent_hand.pop(index)
        self.agent_frame.winfo_children()[len(self.agent_hand)].destroy()

        # moving smoothly the card from the hand to the table
        height = 150
        self.agent_played_card.place(x=250, y=height, anchor="center")
        self.hand_to_table("agent", height)

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
        draw_card = ttk.Button(self.player_frame, style="Card.TButton", image=self.card_images[card_name],
                               command=partial_func)

        # adding the card at the right most place in the player hand
        draw_card.grid(column=len(self.player_hand), row=0, padx=self.card_label_padding,
                       pady=self.card_label_padding)
        self.player_hand.append(card_name)
        if card_name == self.briscola_name:
            self.empty_deck_frame()

    def enable_player_hand(self):
        for i in range(0, len(self.player_frame.winfo_children())):
            self.player_frame.winfo_children()[i]["command"] = self.saved_commands[i]
        self.saved_commands = []

    def disable_player_hand(self):
        self.saved_commands = []
        for child in self.player_frame.winfo_children():
            self.saved_commands.append(child["command"])
            child["command"] = 0

    def release(self):
        with self.cond:
            self.cond.notify()
        self.enable_player_hand()

    def notify_after(self, ms):
        self.disable_player_hand()
        self.root.after(ms, self.release)

    def activate_restart(self, gui_obj):
        partial_func = partial(self.reset_game, gui_obj)
        self.restart_btn["command"] = partial_func

    def deactivate_restart(self):
        self.restart_btn["command"] = 0

    def populate_menu_frame(self):
        """
        Inserts the restart game button into "menu_frame"
        """
        self.restart_btn = ttk.Button(self.menu_frame, text="Restart", command=0, style="Restart.TButton")
        self.restart_btn.grid(column=0, row=0, sticky="NS")

    def populate_deck_frame(self, briscola_name):
        """
        Inserts images of the deck and of the briscola card into the frame "deck_frame".
        """
        empty_label = ttk.Label(self.deck_frame, background="green", width=15)
        briscola_label = ttk.Label(self.deck_frame, style="Card.TButton", image=self.card_images[briscola_name])
        briscola_label.place(relx=0.3, rely=0.4, anchor="center")
        deck_label = ttk.Label(self.deck_frame, style="Retro.TLabel",
                               image=self.card_images["Carte_Napoletane_retro.jpg"])

        empty_label.grid(column=0, row=0)
        deck_label.grid(column=1, row=0, sticky="NS", padx=self.card_label_padding, pady=self.card_label_padding + 10)

    def empty_table_frame(self, winner):
        """
        Removes all the cards from the table frame.

        @param winner: (str) can be "human" or "agent"
        """
        x_player = int(self.player_played_card.place_info()['x'])
        y_player = int(self.player_played_card.place_info()['y'])
        x_agent = int(self.agent_played_card.place_info()['x'])
        y_agent = int(self.agent_played_card.place_info()['y'])
        if winner == 0:
            self.player_played_card.place(x=x_player + 10, y=y_player + 30, anchor="center")
            self.agent_played_card.place(x=x_agent + 10, y=y_agent + 30, anchor="center")
            if y_player > 800:
                self.player_played_card.destroy()
                self.agent_played_card.destroy()
                return
        if winner == 1:
            self.player_played_card.place(x=x_player + 10, y=y_player - 30, anchor="center")
            self.agent_played_card.place(x=x_agent + 10, y=y_agent - 30, anchor="center")
            if y_player < -150:
                self.player_played_card.destroy()
                self.agent_played_card.destroy()
                return
        return self.root.after(1, self.empty_table_frame(winner))

    def empty_deck_frame(self):
        """
        This function loads the image "Deck_Finito.jpg" into the frame "deck_frame".
        """
        for child in self.deck_frame.winfo_children():
            child.destroy()

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

    def reset_log(self):
        """
        Empties the log.
        """
        self.log_text.delete('1.0', END)

    def update_player_score(self, score):
        """
        @param score: (int) score of the player.
        """
        self.player_score["text"] = str(score)

    def update_agent_score(self, score):
        """
        @param score: (int) score of the agent.
        """
        self.agent_score["text"] = str(score)

    def update_deck_count(self, count):
        """
        @param count: (int) number of cards in the deck.
        """
        try:
            self.deck_count_label["text"] = str(count)
        except _tkinter.TclError:
            # this is necessary because if the user repeatedly presses the restart button quickly
            # it may happen that "deck_count_label" is not created in time
            self.deck_count_label = ttk.Label(self.deck_frame, text="", style="Counter.TLabel")
            self.deck_count_label.grid(column=1, row=1)
            self.deck_count_label["text"] = str(count)

    def insert_winner(self, player_name):
        """
        Shows the winner in the table frame through a label.

        @param player_name: (str) the name of the winner.
        """
        for child in self.table_frame.winfo_children():
            child.destroy()
        label = ttk.Label(self.table_frame, text=player_name + " wins!", style="WinnerText.TLabel")
        label.grid(column=0, row=0, sticky="EW")

    def create_main_frames(self):
        """
        Creates main frames, counters and the log.
        """
        self.content.columnconfigure(1, weight=1)
        self.content.rowconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=1)
        self.content.rowconfigure(2, weight=1)

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

        # log
        self.log_text = Text(self.log_frame, background="green", foreground="white", width=35, height=10,
                             padx=self.frame_padding, pady=self.frame_padding, state="disabled")
        self.log_text.grid(column=0, row=0, sticky="EW")
        ys = ttk.Scrollbar(self.log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text['yscrollcommand'] = ys.set
        ys.grid(column=1, row=0, sticky="NS")

        # counters
        style = ttk.Style()
        style.configure("Counter.TLabel", font=("Arial", 15), background="green", foreground="white")

        self.player_score_frame = ttk.Frame(self.content, style="Green.TFrame")
        self.player_score = ttk.Label(self.player_score_frame, style="Counter.TLabel")
        self.player_score_frame.grid(column=2, row=2, sticky="W", padx=30)
        player_score_title = ttk.Label(self.player_score_frame, style="CounterText.TLabel")
        player_score_title.grid(column=0, row=0)
        player_score_title["text"] = "Human score"
        self.player_score.grid(column=0, row=1)

        self.agent_score_frame = ttk.Frame(self.content, style="Green.TFrame")
        self.agent_score = ttk.Label(self.agent_score_frame, style="Counter.TLabel")
        self.agent_score_frame.grid(column=2, row=0, sticky="W", padx=30)
        agent_score_title = ttk.Label(self.agent_score_frame, style="CounterText.TLabel")
        agent_score_title.grid(column=0, row=0)
        agent_score_title["text"] = "Agent score"
        self.agent_score.grid(column=0, row=1)

        self.deck_count_label = ttk.Label(self.deck_frame, text="", style="Counter.TLabel")
        self.deck_count_label.grid(column=1, row=1)

        self.content.columnconfigure(0, weight=0)
        self.content.columnconfigure(1, weight=1)
        self.content.columnconfigure(2, weight=0)
        self.content.columnconfigure(3, weight=0)

    def init_frames(self, gui_obj):
        """
        This method populates the content with initial values, starts the gui for the Briscola game and starts the game.
        """
        try:
            self.new_game_btn.destroy()
        except AttributeError:
            pass
        self.create_main_frames()
        self.populate_menu_frame()
        self.start_game(gui_obj)

    def start_game(self, gui_obj):
        """
        Initializes a new game.
        """
        # initialize the environment
        self.briscola_game = brisc.BriscolaGame(2, self.logger, gui_obj)
        self.briscola_game.reset()

        # initialize the agents
        self.human_agent = HumanAgent()
        agents = [self.human_agent, self.agent]

        # initializing the gui and starting the game
        briscola = self.briscola_game.briscola.name.split()
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

        self.game_thread = threading.Thread(target=brisc.play_episode,
                                            args=(self.briscola_game, agents, gui_obj, False))
        self.game_thread.start()

    def reset_game(self, gui_obj):
        """
        Resets the game and the gui.
        """
        if self.agent_hand and self.player_hand:
            # this means that the game is not finished yet, and it's the user turn
            self.reset = True
            with self.cond:
                self.cond.notify()
        self.player_hand = []
        self.agent_hand = []
        self.content.destroy()
        self.content = ttk.Frame(self.root, padding=(50, 50, 50, 50), style="Green.TFrame")
        self.content.grid(column=0, row=0, sticky="NSEW")
        self.create_main_frames()
        self.populate_menu_frame()
        self.start_game(gui_obj)

    def start_gui(self, gui_obj):
        """
        Starts the gui that is simply a page with 1 button.
        """
        partial_func = partial(self.init_frames, gui_obj)
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)
        self.new_game_btn = ttk.Button(self.content, command=partial_func, text="Start game", style="NewGame.TButton")
        self.new_game_btn.grid(column=0, row=0, sticky="NSEW", padx=350, pady=270)
        self.root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None,
                        help="Provide a trained model path if you want to play against a deep agent", type=str)
    parser.add_argument("--network", default=NetworkTypes.DRQN, choices=[NetworkTypes.DQN, NetworkTypes.DRQN],
                        help="Neural network used for approximating value function")
    flags = parser.parse_args()

    briscola_gui = BriscolaGui()

    if flags.model_dir:
        agent = QAgent(network=flags.network)
        agent.load_model(flags.model_dir)
        agent.make_greedy()
        briscola_gui.agent = agent
    else:
        agent = AIAgent()
        briscola_gui.agent = agent

    logger = BriscolaLogger(BriscolaLogger.LoggerLevels.PVP)
    briscola_gui.logger = logger

    briscola_gui.start_gui(briscola_gui)
