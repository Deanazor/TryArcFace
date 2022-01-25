import os
import csv
import os.path as osp
import random
import argparse

random.seed(6969)
parser = argparse.ArgumentParser()

def issame(name1, name2):
    tmp1 = name1.split(".")[0].split("/")[-1]
    tmp2 = name2.split(".")[0].split("/")[-1]
    return [name1, name2, int(tmp1 == tmp2)]

def create_pairs(lst1, lst2, pair_to=-1):
    if pair_to == -1:
        pairs = [issame(name1, name2) for name1 in lst1 for name2 in lst2]
    else:
        pairs = [issame(name1, name2) for name1, name2 in zip(lst1, lst2)]
        for _ in range(pair_to):
            random.shuffle(lst2)
            pairs.extend([issame(name1, name2) for name1, name2 in zip(lst1, lst2)])
    pairs.sort(key=lambda x: x[0])
    return pairs

def write_csv(output_path, pairs):
    header = ["enroll", "check", "label"]
    with open(output_path, "w") as f:
        writer = csv.writer(f)

        writer.writerow(header)

        writer.writerows(pairs)

        f.close()

if __name__ == "__main__":
    parser.add_argument("--folder_1", required=True, type=str)
    parser.add_argument("--folder_2", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)

    args = parser.parse_args()

    lst1 = [osp.join(args.folder_1, name) for name in os.listdir(args.folder_1)]
    lst2 = [osp.join(args.folder_2, name) for name in os.listdir(args.folder_2)]
    pairs = create_pairs(lst1.copy(), lst2.copy())
    
    write_csv(args.output, pairs)