import os
import pandas as pd
import shutil


def preprocess():
    if os.path.isdir("data/train"):
        print("Looks like already preprocessed")
        return

    os.mkdir("data/train")
    os.mkdir("data/val")

    data_labels, names = get_info()

    counter = 0
    for root, dirs, files in os.walk("data/images"):
        for filename in files:
            subdir = "train"
            if counter % 3 == 0:
                subdir = "val"
            name = names[filename]
            name = name.lower().replace(" ","_")
            dest_folder = f"data/{subdir}/{name}"
            if not os.path.isdir(dest_folder):
                os.mkdir(dest_folder)

            dest = f"{dest_folder}/{counter}.jpg"
            shutil.copyfile(f"data/images/{filename}", dest)
            if counter % 10 == 0:
                print(f"{counter} processed")
            counter += 1


def get_info():
    file = open("data/labels/labels.csv")
    csv_data = pd.read_csv(file)

    df = pd.DataFrame(csv_data)
    data_labels = {}
    names = {}
    for index, row in df.iterrows():
        data_labels[row["Filename"]] = row["Label"]
        names[row["Filename"]] = row["Species"]
    return data_labels, names

if __name__ == "__main__":
    preprocess()