"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
from dotmap import DotMap

# Filenames
TEMPLATE_FILENAME = "template.json"
MY_TEMPLATE_FILENAME = "my_template.json"
EVALUATION_FILENAME = "evaluation.json"
CONFIG_FILENAME = "config.json"

FIELD_LABEL_NUMBER_REGEX = r"([^\d]+)(\d*)"
#
ERROR_CODES = DotMap(
    {
        "MULTI_BUBBLE_WARN": 1,
        "NO_MARKER_ERR": 2,
    },
    _dynamic=False,
)

CUSTOM_BLOCKS = ["QR_CODE", "TEXT_BOX", "NUM_BOX"]

FIELD_TYPES = {
    "QTYPE_INT": {
        "bubbleValues": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "direction": "vertical"
    },
    "QTYPE_INT_FROM_1": {
        "bubbleValues": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        "direction": "vertical",
    },
    "QTYPE_MCQ4": {"bubbleValues": ["A", "B", "C", "D"], "direction": "horizontal"},
    "QTYPE_MCQ5": {
        "bubbleValues": ["A", "B", "C", "D", "E"],
        "direction": "horizontal",
    },
    "QR_CODE" : {
        # "length" : ---      # This is a required field for QR, mention it in template.json based on qr size
    },
    "TEXT_BOX" : {
        # "height" : ---,
        # "length" : ---      # Both are needed for text box
    },
    "NUM_BOX" : {
        # "height" : ---,
        # "length" : ---      # Both are needed for number box
    }
    #
    # You can create and append custom field types here-
    #
}

CHAR_MAP = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
# CNN_MODEL_LOAD_PATH = r"C:\Users\Navaneeth\BTP_1\OMRChecker\src\emnist_cnn_model.h5"
CNN_MODEL_LOAD_PATH = r"C:\Users\Navaneeth\BTP_1\Final_repo\OMR_checker\src\emnist_model.h5"
CNN_MODEL_SAVE_PATH = 'emnist_model.h5'

MNIST_MODEL_LOAD_PATH = r"C:\Users\Navaneeth\BTP_1\Final_repo\OMR_checker\src\mnist_digit_model.h5"
MNIST_MODEL_SAVE_PATH = 'mnist_digit_model.h5'

# TODO: move to interaction.py
TEXT_SIZE = 0.95
CLR_BLACK = (50, 150, 150)
CLR_WHITE = (250, 250, 250)
CLR_GRAY = (130, 130, 130)
CLR_DARK_GRAY = (100, 100, 100)

# TODO: move to config.json
GLOBAL_PAGE_THRESHOLD_WHITE = 200
GLOBAL_PAGE_THRESHOLD_BLACK = 100
