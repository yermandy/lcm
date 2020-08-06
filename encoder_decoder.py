import numpy as np

def create_encoder_decoder():
    encoder = {}
    uid = 0
    for g in ['F', 'M']:
        for a in np.arange(1, 91, 1, dtype=np.uint8):
            encoder[(a, g)] = uid
            uid += 1
    decoder = {v: k for k, v in encoder.items()}
    return encoder, decoder
     

def encode_labels(age, gender, encoder):
    labels = []
    for a, g in zip(age, gender):
        labels.append(encoder[(a, g)])
    return np.array(labels)


def decode_labels(encoded_labels, decoder):
    age, gender = [], []
    for label in encoded_labels:
        a, g = decoder[label]
        age.append(a)
        gender.append(g)
    return np.array(age), np.array(gender)
