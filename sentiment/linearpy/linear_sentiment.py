import sys


def load_data(fname, c):
    # read the processed file and return [[strings], class]
    data = []
    c = int(c)
    with open(fname) as f:
        for line in f:
            fields = line.split()
            try:
                score = float(fields[0])
            except:  # simply ignore them
                continue
            cc = int(score/(1.001/c))
            data.append((fields[1:], cc))
    return data

sys.path.append(".")
from linear_model import model_svm

#### train ####
def main():
    # cmd
    outside_info = {"class": 5, "EXTRA": "-q -t 0"}
    for a in sys.argv[1:]:
        two = a.split(":")
        if(len(two) == 2):
            outside_info[two[0]] = two[1]
        else:
            outside_info["EXTRA"] += " "+a
    # get data
    data_dir = "../data/"
    train_data = load_data(data_dir+"train.txt", outside_info["class"])
    dev_data = load_data(data_dir+"dev.txt", outside_info["class"])
    test_data = load_data(data_dir+"test.txt", outside_info["class"])
    m = model_svm(outside_info)
    m.train_it(train_data, dev_data)
    m.save_it("./")
    print("--test dev:")
    dev_result = m.test_it(dev_data)
    print("--test test:")
    dev_result = m.test_it(test_data)

if __name__ == '__main__':
    main()
