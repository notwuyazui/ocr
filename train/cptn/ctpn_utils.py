#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:05
#
# @Author: Greg Gao(laygin)
#'''
import numpy as np
import cv2
from config import *


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


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


def cal_iou(box1, box1_area , boxes2, boxes2_area): #输入boxes1和boxes2两个点的四个坐标值，以及对应的面积大小
                                                    #输出两个box的IoU值（重叠面积/并集面积）
    """
    box1 [x1,y1,x2,y2]
    boxes2 [Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def cal_overlaps(boxes1, boxes2):   #输出N*M的矩阵，每个box1与每个box2的IoU值
    """
    boxes1 [Nsample,x1,y1,x2,y2]  N*4 anchor
    boxes2 [Msample,x1,y1,x2,y2]  M*4 grouth-box

    """
    #两个box的面积
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])

    #overlaps:N*M
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))

    for i in range(boxes1.shape[0]):
        #对boxes1中的每一个框，计算其与boxes2中所有框的IoU值
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps
    #输出N*M的矩阵，N个anchor，M个ground-box，每个anchor与每个ground-box的IoU值


def bbox_transfrom(anchors, gtboxes):
    """
     compute relative predicted vertical coordinates Vc ,Vh
        with respect to the bounding box location of an anchor
    """
    regr = np.zeros((anchors.shape[0], 2))
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    ha = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h / ha)

    return np.vstack((Vc, Vh)).transpose()


def bbox_transfor_inv(anchor, regr):
    """
        return predict bbox
    """

    Cya = (anchor[:, 1] + anchor[:, 3]) * 0.5
    ha = anchor[:, 3] - anchor[:, 1] + 1

    Vcx = regr[0, :, 0]
    Vhx = regr[0, :, 1]

    Cyx = Vcx * ha + Cya
    hx = np.exp(Vhx) * ha
    xt = (anchor[:, 0] + anchor[:, 2]) * 0.5

    x1 = xt - 16 * 0.5
    y1 = Cyx - hx * 0.5
    x2 = xt + 16 * 0.5
    y2 = Cyx + hx * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox


def clip_box(bbox, im_shape):
    # x1 >= 0
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep


def cal_rpn(imgsize, featuresize, scale, gtboxes):  
    #gtbox：Ground Truth Box真实框
    imgh, imgw = imgsize

    # base_anchor的shape：56000*4，获取了56000个anchor，每个框的左上角右下角共四个坐标数据
    base_anchor = gen_anchor(featuresize, scale)

    # overlaps:56000*200，overlaps[i][j]表示第i个anchor与第j个ground-box的IoU值
    overlaps = cal_overlaps(base_anchor, gtboxes)

    # 构建（初始化的）候选框标签，-1表示不关心，0表示负样本，1表示正样本
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1) #所有数字填充为-1

    # 对每个真实框，取与他IoU最大的anchor的下标
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    
    # 对每个anchor，取与他IoU最大的真实框的下标
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    # 每个anchor的最大的overlaps值
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # IOU > IOU_POSITIVE 记为正样本
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    # IOU <IOU_NEGATIVE  记为负样本
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0
    # 确保每一个ground-box至少有一个anchor被标记为正样本
    labels[gt_argmax_overlaps] = 1

    # 有些候选框越界（超过原图的边框），将其标签设为-1
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgw) |
        (base_anchor[:, 3] >= imgh)
    )[0]
    labels[outside_anchor] = -1

    # 如果正样本个数大于RPN_POSITIVE_NUM，随机选择RPN_POSITIVE_NUM个正样本
    fg_index = np.where(labels == 1)[0]
    # print(len(fg_index))
    if (len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1

    # subsample negative labels
    if not OHEM:
        bg_index = np.where(labels == 0)[0]
        num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
        if (len(bg_index) > num_bg):
            # print('bgindex:',len(bg_index),'num_bg',num_bg)
            labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    # 微调候选框（bounding box），计算每个候选框的偏移量
    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])
    # bbox_targets=[]
    # print(len(labels),len(bbox_targets),len(base_anchor),base_anchor[0],labels[0])

    # 返回[每个候选框的标签，每个候选框的偏移量], 每个候选框的坐标
    return [labels, bbox_targets], base_anchor


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


# for predict
class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class TextLineCfg:
    SCALE = 600
    MAX_SCALE = 1200
    TEXT_PROPOSALS_WIDTH = 16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO = 0.5
    LINE_MIN_SCORE = 0.9
    MAX_HORIZONTAL_GAP = 60
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """

    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index] = True
        return Graph(graph)


class TextProposalConnectorOriented:
    """
        Connect text proposals into text lines
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        # len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals:boxes

        """
        # tp=text proposal
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)  # 首先还是建图，获取到文本行由哪几个小框构成

        text_lines = np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]  # 每个文本行的全部小框
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 求每一个小框的中心x，y坐标
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            z1 = np.polyfit(X, Y, 1)  # 多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）

            x0 = np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
            x1 = np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # 小框宽度的一半

            # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))  # 求全部小框得分的均值作为文本行的均值

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
            text_lines[index, 4] = score  # 文本行得分
            text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 做补偿
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs

