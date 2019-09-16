from matplotlib import pyplot as plt


def read_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        content = [eval(line.replace("\n", "")) for line in file.readlines()]
        return content

def get_train_data(data):
    assert isinstance(data, list), "``data`` should be list type"
    epoch = []
    acc = []
    loss = []
    for line in data:
        if line["mode"] == "train":
            epoch.append(line["epoch"])
            acc.append(line["accuracy"])
            loss.append(line["loss"])
    return {"epoch": epoch, "accuracy": acc, "loss": loss}

def get_val_data(data):
    assert isinstance(data, list), "``data`` should be ``list`` type"
    epoch = []
    acc = []
    loss = []
    for line in data:
        if line["mode"] == "test":
            epoch.append(line["epoch"])
            acc.append(line["accuracy"])
            loss.append(line["loss"])
    return {"epoch": epoch, "accuracy": acc, "loss": loss}    

def get_best_val_acc(data):
    assert isinstance(data, list), "``data`` must be ``list`` type"
    return max(data)

def show(data):
    assert isinstance(data, dict), "``data`` should be ``dict`` type"
    _, axs = plt.subplots(1, 2,figsize=(16, 4), sharey=False)
    axs[0].plot(data["epoch"], data["accuracy"])
    axs[0].set_title("accuracy")

    axs[1].plot(data["epoch"], data["loss"])
    axs[1].set_title("loss")
    plt.show()

if __name__ == "__main__":
    path = "F:/mixed-densenet/state/state_full_in_loop.txt"
    data = read_txt(path)
    train_data = get_train_data(data)
    print("best train accuracy:")
    print(get_best_val_acc(train_data["accuracy"]))
    val_data = get_val_data(data)
    print("best val accuracy:")
    print(get_best_val_acc(val_data["accuracy"]))
    show(train_data)
    show(val_data)
    