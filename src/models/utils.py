import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    r"""Sets model to feature extraction mode or not. If **feature_extracting** is True, the gradients are frozen in the
    model. Else, the gradients are activated.

    Inputs:
        - **model** (PyTorch model): Model to set
        - **feature_extracting** (bool): If true, freezes model gradients. If not, activates model gradients.

    """

    for param in model.parameters():
        param.requires_grad = not feature_extracting


classes = ['drink water', 'eat meal/snack',
           'brushing teeth', 'brushing hair',
           'drop', 'pickup',
           'throw', 'sitting down',
           'standing up', 'clapping',
           'reading', 'writing',
           'tear up paper', 'wear jacket',
           'take off jacket', 'wear a shoe',
           'take off a shoe', "wear on glasses",
           'take off glasses', 'put on a hat/cap',
           'take off a hat/cap', 'cheer up',
           'hand waving', 'kicking something',
           'reach into pocket', 'hopping (one foot jumping)',
           'jump up', 'make a phone call/answer phone',
           'playing with phone/tablet', 'typing on keyboard',
           'pointing to something with finger', 'taking a selfie',
           'check time (watch)', 'rub hands together',
           'nod head/bow', 'shake head',
           'wipe face', 'salute',
           'put the palms together', 'cross hands in front',
           'sneeze/couggh', 'staggering',
           'falling', 'touch head',
           'touch chest', 'touch back',
           'touch neck', 'nausea or vomiting',
           'use a fan', 'punching/slapping other person',
           'kicking other person', 'pushing other person',
           'pat on back of other person', 'point finger at other person',
           'hugging other person', 'giving something to other person',
           'touch other person pocket', 'handshaking',
           'walk toward other', 'walk apart from each other']
