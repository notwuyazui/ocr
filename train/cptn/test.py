import numpy as np
def gen_anchor(featuresize, scale): #scale=16
    """
        gen base anchor from feature map [HXW][9][4]
        reshape  [HXW][9][4] to [HXWX9][4]
    """
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)
    #after reshape, size:10*1

    base_anchor = np.array([0, 0, 15, 15])  #基础框
    # 计算中心点
    xt = (base_anchor[0] + base_anchor[2]) * 0.5    
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    #还原出x1 y1 x2 y2（size：10*1）
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    #base_anchor:10个框的左上角和右下角的坐标
    base_anchor = np.hstack((x1, y1, x2, y2))       #按水平叠放成一个数组，size：10*4*1

    h, w = featuresize  #56，100
    shift_x = np.arange(0, w) * scale   #一个点相当于16个点，还原成原始图像坐标
    shift_y = np.arange(0, h) * scale
    #x:[0 16 32 ... 1584],y:[0 16 32 ... 896]
    
    # apply shift
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])
    #anchor里面5600个array，每个array是1*10*4十个框，每个框四个坐标点
    #np.array(anchor)的shape:[5600, 10, 4]
    return np.array(anchor).reshape((-1, 4))
    #返回56000*4,56000个框，每个框的四个坐标

array=gen_anchor( (56,100), 16)
print(array.shape)