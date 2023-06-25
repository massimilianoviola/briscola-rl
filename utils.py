class BriscolaLogger:
    """Adjust verbosity on four levels as needed."""

    class LoggerLevels:
        DEBUG = 0
        PVP = 1
        TRAIN = 2
        TEST = 3

    def __init__(self, verbosity=3):
        self.configure_logger(verbosity)

    def configure_logger(self, verbosity):

        self.verbosity = verbosity

        if self.verbosity > self.LoggerLevels.DEBUG:
            self.DEBUG = lambda *args: None
        else:
            self.DEBUG = print

        if self.verbosity > self.LoggerLevels.PVP:
            self.PVP = lambda *args: None
        else:
            self.PVP = print

        if self.verbosity > self.LoggerLevels.TRAIN:
            self.TRAIN = lambda *args: None
        else:
            self.TRAIN = print

        self.TEST = print


class CardsEncoding:
    HOT_ON_DECK = 'hot_on_deck'
    HOT_ON_NUM_SEED = 'hot_on_num_seed'


class CardsOrder:
    APPEND = 'append'
    REPLACE = 'replace'
    VALUE = 'value'


class NetworkTypes:
    DQN = 'dqn'
    DRQN = 'drqn'
    AC = 'actor_critic'


class PlayerState:
    HAND_PLAYED_BRISCOLA = 'hand_played_briscola'
    HAND_PLAYED_BRISCOLASEED = 'hand_played_briscolaseed'
    HAND_PLAYED_BRISCOLA_HISTORY = 'hand_played_briscola_history'


def convert_to_binary_list(decimal_number, max_bits):
    if decimal_number > 2 ** max_bits - 1:
        raise ValueError
    binary_number = bin(decimal_number)[2:]  # Convert decimal to binary and remove the '0b' prefix
    # Pad the binary number with leading zeros to ensure a length of 7 bits
    binary_number = binary_number.zfill(max_bits)
    # Store each bit in a list
    binary_list = [int(bit) for bit in binary_number]
    return binary_list


def overwrite_values(start_index, target_vector, values_list):
    target_vector[start_index:start_index + len(values_list)] = values_list