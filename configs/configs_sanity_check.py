import yaml

def f():
    filename="voc_resnet50d_30epochs.yaml"
    with open(filename) as file:
        dic=yaml.full_load(file)
        print(dic)

if __name__=="__main__":
    f()
