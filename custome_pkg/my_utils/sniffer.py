import os
import copy
import re


class Sniffer:
    def __init__(self):
        self.all_file_path_list = None
        pass

    def sniff_file(self, directory_path, pattern, max_deep=-1):
        """
        Get all files in the directory_path that match the pattern
        :param directory_path:
        :param pattern:
        :param max_deep:
        :return:
        """
        ans = list()
        self.all_file_path_list = list()
        self.get_all_by_recursion(directory_path, deep=0, max_deep=max_deep)
        path_list = copy.deepcopy(self.all_file_path_list)
        for path in path_list:
            path_ = re.split(pattern="/", string=path)
            filename = path_[len(path_) - 1]
            if re.findall(pattern=pattern, string=filename):
                ans.append(path)
        return ans

    def get_all_by_recursion(self, directory_path, deep, max_deep):
        """
        # Recursively get all files in the directory_path up to a maximum depth of max_deep, and save them to self.all_file_path_list
        :param directory_path:
        :param deep:
        :param max_deep:
        :return:
        """
        if directory_path[len(directory_path) - 1] == "/":
            directory_path = directory_path[0:-1]
        if os.path.isdir(directory_path) and (max_deep < 0 or deep < max_deep):
            filename_list = os.listdir(path=directory_path)
            for filename in filename_list:
                self.get_all_by_recursion(directory_path + "/" + filename, deep + 1, max_deep)
        else:
            self.all_file_path_list.append(directory_path)


if __name__ == "__main__":
    pass
