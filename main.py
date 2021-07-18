import numpy as np
import tensorflow as tf
import dplython

# Result column, 1 = win, 0 = loss
def main():
    data = np.genfromtxt("data/2021_LoL_esports_match_data_from_OraclesElixir_20210515.csv", delimiter=",",names=True, dtype=None, comments=None, skip_header=0)
    # Use a breakpoint in the code line below to debug your script.

    print(data["position" == "team"])

    filter_data = list(filter(lambda row: row[13] == "", data))
    np.savetxt
    print(" ")







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
