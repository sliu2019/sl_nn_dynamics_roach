import os
import numpy


if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.getcwd(), "../data_collection/"))
    lst = []
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            tmp = os.path.join(subdir, file)
            lst.append(tmp)
    lst.sort()
    # for i in lst:
    # 	print i
    print len(lst)