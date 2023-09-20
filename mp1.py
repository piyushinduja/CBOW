import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
args = parser.parse_args()
output_dir = args.output_dir

VOCAB_PATH = output_dir + r"/vocab.txt"
DATA_PATH = output_dir + r"/data"
MODELS_PATH = output_dir + r"/models"
EMBEDDINGS_PATH = output_dir + r"/embeddings"

def get_vocab():
    with open(VOCAB_PATH, 'r', encoding='utf8') as vocab:
        v0 = []
        v1 = []
        for line in vocab:
            p = line.split()
            v0.append(int(p[0]))
            v1.append(p[1])
    return v0, v1


# Data Preprocessing
def create_dataset(path, files, C_SIZE):

    main_data = []
    c_data = []
    v0, v1 = get_vocab()
    word_to_i = {word: i for i, word in enumerate(v1)}
    i_to_word = {i: word for i, word in enumerate(v1)}
    for k in range(1, files+1):

        data = open(f'{DATA_PATH}/{path}/{k}.txt', 'r', encoding='utf8').read().split()

        for i in range(data.__len__()):
            if data[i] not in v1:
                data[i] = '[UNK]'

        # Contexts
        for i in range(0, len(data)):

            if i < C_SIZE:
                pad = ['[PAD]' for i in range(i, C_SIZE)]
                context = (pad + [j for j in data[0:i]] + [j for j in data[i + 1:i + C_SIZE + 1]])
                c_data.append(context)

            elif i >= len(data) - C_SIZE:
                pad = ['[PAD]' for i in range(i - (len(data) - C_SIZE) + 1)]
                context = ([j for j in data[i - C_SIZE:i]] + [j for j in data[i + 1:len(data)]] + pad)
                c_data.append(context)

            else:
                context = ([j for j in data[i - C_SIZE:i]] + [j for j in data[i + 1:i + C_SIZE + 1]])
                c_data.append(context)

            target = data[i]
            c_id = [word_to_i[w] for w in context]
            t_id = word_to_i[target]
            main_data.append((c_id, t_id))

    return main_data, word_to_i, i_to_word


# Model
class CBOWModel(nn.Module):

    def __init__(self, v_size, e_dim):
        super().__init__()
        self.embeddings = nn.Embedding(v_size, e_dim)
        self.linear = nn.Linear(e_dim, v_size, bias=False)

    def forward(self, x):
        embeds = self.embeddings(x).mean(1).squeeze(1)
        out = self.linear(embeds)
        return out


def closer_pairs(w1, w2, w3, w4, vocab_dict):
    print(f"[{w1}, {w2}] or [{w3}, {w4}]")
    v1 = vocab_dict[w1]
    v2 = vocab_dict[w2]
    v3 = vocab_dict[w3]
    v4 = vocab_dict[w4]
    sim1 = cosine_similarity([v1], [v2])
    sim2 = cosine_similarity([v3], [v4])
    print(f"Similarity of {w1} and {w2}: ", sim1)
    print(f"Similarity of {w3} and {w4}: ", sim2)
    if sim1 > sim2:
        print(f"Pair [{w1}, {w2}] is closer.")
    elif sim2 > sim1:
        print(f"Pair [{w3}, {w4}] is closer.")
    else:
        print("Both the pairs are equally closer.")


def analogous_word(w1, w2, w3, vocab_dict):

    maximum_similarity = -99999
    w4 = None
    words = vocab_dict.keys()
    va, vb, vc = vocab_dict[w1], vocab_dict[w2], vocab_dict[w3]

    for i in words:
        if i in [w1, w2, w3]:
            continue
        w_vec = vocab_dict[i]
        similarity = cosine_similarity([[b - a for a, b in zip(va, vb)]], [[d - c for d, c in zip(vc, w_vec)]])

        if similarity > maximum_similarity:
            maximum_similarity = similarity
            w4 = i

    return w4


def main():
    # Creating dataset and data loader
    EMBEDDING_SIZE = 100
    EPOCHS = 10
    C_SIZE = 5
    LEARNING_RATES = [0.01, 0.001, 0.0001]
    v0, v1 = get_vocab()
    V_SIZE = v1.__len__()  # 18061

    # Making training and dev dataset and data loaders

    main_data, word_2_i, i_2_word = create_dataset('train', 30, C_SIZE)
    c_data = torch.tensor([main_data[i][0] for i in range(main_data.__len__())])
    t_data = torch.tensor([main_data[i][1] for i in range(main_data.__len__())])
    train_dataset = TensorDataset(c_data, t_data)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    dev_data, dev_w_2_i, dev_i_2_w = create_dataset('dev', 5, C_SIZE)
    dev_c_data = torch.tensor([dev_data[i][0] for i in range(dev_data.__len__())])
    dev_t_data = torch.tensor([dev_data[i][1] for i in range(dev_data.__len__())])
    dev_dataset = TensorDataset(dev_c_data, dev_t_data)
    dev_dataloader = DataLoader(dev_dataset, batch_size=64)

    model = CBOWModel(e_dim=EMBEDDING_SIZE, v_size=V_SIZE)

    best_loss = [10000.000, 0]
    for lr in LEARNING_RATES:
        print("Learning rate: ", lr)

        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_perf_dict = {"epoch": 0, "loss": 1000000, "model": {}, "optimizer": {}}

        # Training loop
        for i in range(EPOCHS):

            # Training loop
            train_losses = []
            for context, target in tqdm(train_dataloader):
                model.train()
                pred = model(context)
                loss = loss_func(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            print(f'Epoch {i + 1}: Training Loss {np.average(train_losses)}')

            # Evaluation loop
            dev_losses = []
            gold_labels = []
            pred_labels = []
            for context, target in tqdm(dev_dataloader):
                model.eval()
                with torch.no_grad():
                    out = model(context)
                    pred = torch.argmax(out, dim=1)
                    loss1 = loss_func(out, target.long())

                    pred_labels.extend(pred.tolist())
                    gold_labels.extend(target.tolist())

                    dev_losses.append(loss1.item())

            dev_loss = np.average(dev_losses)

            print(f'Epoch {i + 1}: Dev Loss {dev_loss}')

            if dev_loss > best_perf_dict["loss"]:
                best_perf_dict["epoch"] = i+1
                best_perf_dict["loss"] = np.average(dev_losses)
                best_perf_dict["model"] = model.state_dict()
                best_perf_dict["optimizer"] = optimizer.state_dict()

        torch.save({
            "model_param": best_perf_dict["model"],
            "optim_param": best_perf_dict["optimizer"],
            "epoch": best_perf_dict["epoch"],
            "loss": best_perf_dict["loss"]
        }, f"{MODELS_PATH}/best/lr_{lr}/")

        print(f"Dev Loss with learning rate {lr} is {best_perf_dict['loss']}")
        if best_loss[0] > best_perf_dict['loss']:
            best_loss[0] = best_perf_dict['loss']
            best_loss[1] = lr

    # Making embeddings file
    optimal_lr = best_loss[1]
    model_path = f"{MODELS_PATH}/best/lr_{optimal_lr}"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_param"])
    optimizer1 = torch.optim.Adam(model.parameters(), lr=optimal_lr)
    optimizer1.load_state_dict(checkpoint["optim_param"])
    lin_weights = model.state_dict()['linear.weight'].tolist()
    emb_weights = model.state_dict()['embeddings.weight'].tolist()

    with open(f'{EMBEDDINGS_PATH}/embed.txt', 'a', encoding='utf8') as embed:

        for i in range(v0.__len__()):
            word = i_2_word[i]
            weights = emb_weights[i]
            str1 = ""
            for j in weights:
                str1 = str1 + str(j) + " "
            embed.write(f"{word} {str1} \n")

    # Q-2)
    file = open(f'{EMBEDDINGS_PATH}/embed.txt', 'r', encoding='utf8')
    data = file.read()
    file.close()

    list1 = data.split('\n')
    list1.pop()
    vocab_dict = {}
    for i in range(len(list1)):
        str1 = list1[i]
        list2 = str1.split(' ')
        list2.pop()
        list2.pop()
        vocab_dict[list2[0]] = [float(list2[j]) for j in range(1, len(list2))]

    print("Q-2) a) Which pairs are closer? \n")
    closer_pairs('cat', 'tiger', 'plane', 'human', vocab_dict)
    print('\n')
    closer_pairs('my', 'mine', 'happy', 'human', vocab_dict)
    print('\n')
    closer_pairs('happy', 'cat', 'king', 'princess', vocab_dict)
    print('\n')
    closer_pairs('ball', 'racket', 'good', 'ugly', vocab_dict)
    print('\n')
    closer_pairs('cat', 'racket', 'good', 'bad', vocab_dict)
    print('\n')

    print("Q-2) b) Analogies:\n")
    print("1) king: queen, man:", analogous_word('king', 'queen', 'man', vocab_dict))
    print("2) king: queen, prince:", analogous_word('king', 'queen', 'prince', vocab_dict))
    print("3) king: man, queen:", analogous_word('king', 'man', 'queen', vocab_dict))
    print("4) woman: man, princess:", analogous_word('woman', 'man', 'princess', vocab_dict))
    print("5) prince: princess, man:", analogous_word('prince', 'princess', 'man', vocab_dict))


if __name__ == "__main__":
    main()
