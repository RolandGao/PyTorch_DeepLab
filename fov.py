import torchvision

def field_of_vision(fs):
    k=1
    s=1
    for _k,_s in fs: # kernel size and stride
        k=k+(_k-1)*s
        s=s*_s
    return k,s

def staged_net(ds):
    fs=[]
    for d in ds:
        for i in range(d):
            if i==0:
                fs.append([3,2])
            else:
                fs.append([3,1])
    return fs


def test1():
    regnets=[
        [1,1,4,7],
        [1,2,7,12],
        [1,3,5,7],
        [1,3,7,5],
        [2,4,10,2],
        [2,6,15,2],
        [2,5,14,2],
        [2,4,10,1],
        [2,5,15,1],
        [2,5,11,1],
        [2,6,13,1],
        [2,7,13,1]
    ]
    for ds in regnets:
        fs=staged_net(ds)
        fs=[(3,2)]+fs
        k,s=field_of_vision(fs)
        print(k,s)
def test2():
    resnets=[
        [2, 2, 2, 2],
        [3, 4, 6, 3],
        [3, 4, 6, 3],
        [3, 4, 23, 3],
        [3, 8, 36, 3]
    ]
    for ds in resnets:
        ds[0]=ds[0]+1
        fs=staged_net(ds)
        fs=[(7,2)]+fs
        #fs=[(3,2),(3,1),(3,1)]+fs
        k,s=field_of_vision(fs)
        print(k,s)
def test3():
    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    fs=[]
    mobilenetv2=[2,3,7,4]
    mobilenetv3_large_fs=[
        (3,2),(3,1),(3,2),(3,1),(5,2),(5,1),(5,1),(3,2),
        (3,1),(3,1),(3,1),(3,1),(3,1),(5,2),(5,1),(5,1)]
    fs=staged_net(mobilenetv2)
    fs=[(3,2),(3,1)]+fs
    k,s=field_of_vision(fs)
    print(k,s)
    k,s=field_of_vision(mobilenetv3_large_fs)
    print(k,s)


if __name__=="__main__":
    test2()
    torchvision.models.resnet50()
    #model=torchvision.models.mobilenet_v2()
    # fs=[(3,2),(1,1),(3,2),(1,1)]
    # k,s=field_of_vision(fs)
    # print(k,s)
