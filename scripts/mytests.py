import os
import numpy


if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.getcwd(), "../data_collection/"))
    lst = []
    task_type = ["carpet"]
    months = ['02']
    for subdir, dirs, files in os.walk(data_path):
        l = subdir.split("/")[-1].split("_")
        if len(l) >= 3:
            surface = l[0]
            month = l[2]
            if (surface in task_type or task_type == "all") and month in months:
                print month
                for file in files:
                    tmp = os.path.join(subdir, file)
                    lst.append(tmp)
    lst.sort()
    # for i in lst:
    # 	print i
    print len(lst)/2