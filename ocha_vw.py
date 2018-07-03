#coding:UTF-8

import sys
from collections import namedtuple

#change Mecab(ochasen-mode) format to Vowpal Wabbit format
#+1 |token a b ... |yomi |baseform |pos

def item_print(items):
    out = str()
    for item in items:
        out += item
    return out

def formatize(doc):
    tokens = []
    yomis = []
    baseforms = []
    poses = []
    for line in doc:
        if line != "\n":
            if "EOS" in line[:-1]:
                print("|token "+str(item_print(tokens))+"|yomi "+str(item_print(yomis))+"|baseform "+str(item_print(baseforms))+"|pos "+str(item_print(poses)))
                tokens = []
                yomis = []
                baseforms = []
                poses = []
            else:
                newline = []
                for item in line.split("\t"):
                    if item == '':
                        item = "_"
                    newline.append(item)
                token, yomi, baseform, pos1, pos2, pos3 = [item for item in newline]
                tokens.append(token+" ")
                yomis.append(yomi+" ")
                baseforms.append(baseform+" ")
                pos = pos1+"-"+pos2+"-"+pos3[:-1]
                poses.append(pos+" ")
    return

def main():
    argv = sys.argv
    in_line = argv[1]
    with open(in_line,"r") as file1:
        doc = file1.readlines()
    formatize(doc)
if __name__ == '__main__':
    main()
