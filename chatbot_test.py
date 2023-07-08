import json
import random
import numpy as np
from termcolor import colored

import trax
from trax import layers as tl
from trax.supervised import training
import tkinter as tk


# filename of the MultiWOZ dialogue dataset
DATA_FILE = 'data.json'

# data directory
DATA_DIR = './data'

# dictionary where we will load the dialogue dataset
DIALOGUE_DB = {}

# vocabulary filename
VOCAB_FILE = 'en_32k.subword'

# vocabulary file directory
VOCAB_DIR = 'data/vocabs'

# delimiters used during training
delimiter_1 = 'Person 1: '
delimiter_2 = 'Person 2: '


def ReformerLM(vocab_size=33000, n_layers=2, mode='train', attention_type=tl.SelfAttention):
    # initialized instance of Trax's ReformerLM class
    model = tl.Serial(
        trax.models.reformer.ReformerLM(
            # vocab size
            vocab_size=vocab_size,
            # number of layers
            n_layers=n_layers,
            # mode
            mode=mode,
            # attention type
            attention_type=attention_type
        )
        , tl.LogSoftmax()
    )
    return model

def tokenize(sentence, vocab_file, vocab_dir):
    return list(trax.data.tokenize(iter([sentence]), vocab_file=vocab_file, vocab_dir=vocab_dir))[0]

def detokenize(tokens, vocab_file, vocab_dir):
    return trax.data.detokenize(tokens, vocab_file=vocab_file, vocab_dir=vocab_dir)


def ReformerLM_output_gen(ReformerLM, start_sentence, vocab_file, vocab_dir, temperature, tokenize=tokenize):

    # input tokens created using  the tokenize function
    input_tokens = tokenize(start_sentence, vocab_file=vocab_file, vocab_dir=vocab_dir)

    # batch dimension added to array. Converted from (n,) to (1, n)
    input_tokens_with_batch = np.array(input_tokens)[None, :]

    # autoregressive_sample_stream function from trax
    output_gen = trax.supervised.decoding.autoregressive_sample_stream(
        # model
        ReformerLM,
        # inputs are tokens with batch dimension
        inputs=input_tokens_with_batch,
        # temperature
        temperature=temperature
    )

    return output_gen


# defined `predict_mem_len` and `predict_drop_len` of tl.SelfAttention
def attention(*args, **kwargs):
    # number of input positions to remember in a cache when doing fast inference.
    kwargs['predict_mem_len'] = 120
    # number of input elements to drop once the fast inference input cache fills up.
    kwargs['predict_drop_len'] = 120
    # attention layer returned with the parameters defined above
    return tl.SelfAttention(*args, **kwargs)

# defined model using the ReformerLM function implemented earlier.
model = ReformerLM(
    vocab_size=33000,
    n_layers=6,
    mode='predict',
    attention_type=attention,
)

# defined input signature so model can be initialized. shape will be (1, 1) and the data type is int32.
shape11 = trax.shapes.ShapeDtype((1, 1), dtype=np.int32)

# initialized from file
model.init_from_file('chatbot_model1.pkl.gz',
                     weights_only=True, input_signature=shape11)


def generate_response(ReformerLM, start_sentence, temperature):

    #looking for only last two interactions due to memory problems
    if start_sentence.count(delimiter_1) >= 2:
        start_sentence = start_sentence[start_sentence.rfind(delimiter_1,0,start_sentence.rfind(delimiter_1)):]

    result=[]

    # calls the output generator implemented earlier
    output = ReformerLM_output_gen(ReformerLM, start_sentence, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR,
                                   temperature=temperature)

    # loop below yields response

    for o in output:

        result.append(o)
        sentence = detokenize(np.concatenate(result, axis=0), vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)

        if sentence.endswith(delimiter_1):
            sentence = sentence.split(delimiter_1)[0]
            return sentence


def displaying_response():
    question_sentence = question.get("0.1","end").strip()
    sample_for_function = dialog.get("0.1","end").replace("\n"," ").strip() + delimiter_1 + question_sentence + " " + delimiter_2
    response = generate_response(ReformerLM=model, start_sentence=sample_for_function, temperature=0.2)
    full_dialog = sample_for_function + response

    split_list_d1 = full_dialog.split(delimiter_1)
    dd=""
    for sublist in split_list_d1[1:]:
        split_list_d2 = sublist.split(delimiter_2)
        dd = dd + delimiter_1 + split_list_d2[0] + "\n"

        if len(split_list_d2) > 1:
            dd = dd + delimiter_2 + split_list_d2[1] + "\n"
    dialog.delete("0.1", "end")
    dialog.insert("0.1",dd)

### interface
root = tk.Tk()
root.title("Chatbot")
root.geometry("600x300")

dlabel = tk.Label(root, text="Dialog")
dlabel.pack()

dialog = tk.Text(root,height=10, width=70)
dialog.pack()

qlabel = tk.Label(root, text="Question")
qlabel.pack()

question = tk.Text(root,height=1, width=70)
question.pack()

space_label = tk.Label(root, text=" ")
space_label.pack()

bt = tk.Button(root, text="Generate response", command=displaying_response)
bt.pack()

root.mainloop()
