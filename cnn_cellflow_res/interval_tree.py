# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-27
description:
    区间树算法
"""


# 节点类
class Node:
    def __init__(self, right, left, p, color, inter, maxx):
        """
        :param right: 右子节点
        :param left: 左子节点
        :param p: 父节点
        :param color: 该节点颜色，因为区间树是红黑树
        :param inter: 区间范围，一个Inter类
        :param maxx: 区间的极大值
        """
        self.key = inter.low
        self.right = right
        self.left = left
        self.p = p
        self.color = color
        self.inter = inter
        self.maxx = maxx


# 代表区间的类
class Inter:
    def __init__(self, low, high):
        self.low = low
        self.high = high


class tree:
    def __init__(self, root, nil):
        self.root = root
        self.nil = nil

    def tree_insert(self, z):
        y = self.nil
        x = self.root
        while x != self.nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.p = y
        if y == self.nil:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = "RED"
        z.maxx = max(z.inter.high, z.left.maxx, z.right.maxx)
        # 红黑树性质维护
        self.rb_insert_fixup(z)
        # 更新父结点直到根结点的maxx
        while z.p != self.nil:
            z.p.maxx = max(z.p.maxx, z.maxx)
            z = z.p

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.nil:
            y.left.p = x
        y.p = x.p
        if x.p == self.nil:
            self.root = y
        elif x == x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y
        # 左旋导致两个结点的max属性改变，更新如下
        y.maxx = x.maxx
        x.maxx = max(x.left.maxx, x.right.maxx, x.inter.high)

    def right_rotate(self, y):
        x = y.left
        y.left = x.right
        if x.right != self.nil:
            x.right.p = y
        x.p = y.p
        if y.p == self.nil:
            self.root = x
        elif y == y.p.left:
            y.p.left = x
        else:
            y.p.right = x
        x.right = y
        y.p = x
        # 右旋导致两个结点的max属性改变，更新如下
        x.maxx = y.maxx
        y.maxx = max(y.right.maxx, y.left.maxx, y.inter.high)

    def rb_insert_fixup(self, z):
        while z.p.color == "RED":
            if z.p == z.p.p.left:
                y = z.p.p.right
                if y.color == "RED":
                    z.p.color = "BLACK"
                    y.color = "BLACK"
                    z.p.p.color = "RED"
                    z = z.p.p
                else:
                    if z == z.p.right:
                        z = z.p
                        self.left_rotate(z)
                    z.p.color = "BLACK"
                    z.p.p.color = "RED"
                    self.right_rotate(z.p.p)
            else:
                y = z.p.p.left
                if y.color == "RED":
                    z.p.color = "BLACK"
                    y.color = "BLACK"
                    z.p.p.color = "RED"
                    z = z.p.p
                else:
                    if z == z.p.left:
                        z = z.p
                        self.right_rotate(z)
                    z.p.color = "BLACK"
                    z.p.p.color = "RED"
                    self.left_rotate(z.p.p)
        self.root.color = "BLACK"

    def inorder_tree_walk(self, x):
        if x != self.nil:
            self.inorder_tree_walk(x.left)
            print(x.key)
            self.inorder_tree_walk(x.right)

    def tree_search(self, x, k):
        if x == self.nil or k == x.key:
            return x
        if k < x.key:
            return self.tree_search(x.left, k)
        else:
            return self.tree_search(x.right, k)

    def rb_transplant(self, u, v):
        if u.p == self.nil:
            self.root = v
        elif u == u.p.left:
            u.p.left = v
        else:
            u.p.right = v
        v.p = u.p

    def tree_minimum(self, x):
        while x.left != self.nil:
            x = x.left
        return x

    def rb_delete(self, z):
        y = z
        y_original_color = y.color
        if z.left == self.nil:
            x = z.right
            self.rb_transplant(z, z.right)
        elif z.right == self.nil:
            x = z.left
            self.rb_transplant(z, z.left)
        else:
            y = self.tree_minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.p == z:
                x.p = y
            else:
                self.rb_transplant(y, y.right)
                y.right = z.right
                y.right.p = y
            self.rb_transplant(z, y)
            y.left = z.left
            y.left.p = y
            y.color = z.color
            if y_original_color == "BLACK":
                self.rb_delete_fixup(x)

    def rb_delete_fixup(self, x):
        while x != self.root and x.color == "BLACK":
            if x == x.p.left:
                w = x.p.right
                if w.color == "RED":
                    w.color = "BLACK"
                    x.p.color = "RED"
                    self.left_rotate(x.p)
                    w = x.p.right
                if w.left.color == "BLACK" and w.right.color == "BLACK":
                    w.color = "RED"
                    x = x.p
                else:
                    if w.right.color == "BLACK":
                        w.left.color = "BLACK"
                        w.color = "RED"
                        self.right_rotate(w)
                        w = x.p.right
                    w.color = x.p.color
                    x.p.color = "BLACK"
                    w.right.color = "BLACK"
                    self.left_rotate(x.p)
                    x = self.root
            else:
                w = x.p.left
                if w.color == "RED":
                    w.color = "BLACK"
                    x.p.color = "RED"
                    self.right_rotate(x.p)
                    w = x.p.left
                if w.right.color == "BLACK" and w.left.color == "BLACK":
                    w.color = "RED"
                    x = x.p
                else:
                    if w.left.color == "BLACK":
                        w.right.color = "BLACK"
                        w.color = "RED"
                        self.left_rotate(w)
                        w = x.p.left
                    w.color = x.p.color
                    x.p.color = "BLACK"
                    w.left.color = "BLACK"
                    self.right_rotate(x.p)
                    x = self.root
        x.color = "BLACK"

    def print_tree(self, z):
        if z != self.nil:
            print(z.key, z.color, "[", z, inter.low, ",", z.inter.high, "]", ":", end='')
            print("( ", end='')
            print(z.left.key, z.left.color, "   ", end='')
            print(z.right.key, z.right.color, end='')
            print(" )", end='')

    def interval_search(self, i):
        x = self.root
        while x != self.nil and (i.high < x.inter.low or x.inter.high < i.low):
            if x.left != self.nil and x.left.maxx >= i.low:
                x = x.left
            else:
                x = x.right
        return x


class _Node:
    def __init__(self, center, interval_starts, interval_ends, left, right):
        """
        :param center: 当前节点的父节点的区间范围。
        :param by_low: 当前节点的所有区间的下界的排序。
        :param by_high: 当前节点的所有区间的上界的排序。
        :param left: 左子节点的引用。
        :param right: 右子节点的引用。
        """
        self.center = center
        self.interval_starts = interval_starts
        self.interval_ends = interval_ends
        self.left = left
        self.right = right


class IntervalTree:
    def __init__(self):
        """
        """
        pass

    def interval_tree(self, intervals):
        """
        构建区间树。
        :return:
        """
        assert intervals == sorted(intervals)
        if not intervals:
            return None
        # 选取center（作为根节点即其他中间节点），center是中间的样本的其实位点
        center = intervals[len(intervals) // 2][0]

        # 通过center将全部样本分为L、R和C三组。L组的区间都在center的左侧；R组都在center右侧；C组则包含中间值center
        L, R, C = [], [], []
        for i in intervals:
            if i[1] < center:
                L.append(i)
            elif i[0] > center:
                R.append(i)
            else:
                C.append(i)
        interval_starts = sorted((i[0], i) for i in C)
        interval_ends = sorted((i[1], i) for i in C)
        # 递归生成左树和右树
        left = self.interval_tree(L)
        right = self.interval_tree(R)

        # 表面上看是返回一个节点，但是该节点包含的信息可以延伸至整棵树
        return _Node(center, interval_starts, interval_ends, left, right)

    def intervals_containing(self, tree, position):
        """
        查询某个位点的所在区间。
        :param tree: 区间树对象
        :param position: 位点的坐标，整数型。
        :return:
        """
        if tree is None:
            return False
        # 如果位点小于根节点的center，则递归的在左子树中搜索
        if position < tree.center:
            res = self.intervals_containing(tree.left, position)
        else:
            # 是否被当前节点的峰包含
            covered_by_cur_node = self.in_peak(tree.interval_starts, tree.interval_ends, position)
            if not covered_by_cur_node:
                res = self.intervals_containing(tree.right, position)
            else:
                return covered_by_cur_node
        return res

    def in_peak(self, interval_starts, interval_ends, position):
        """
        :param interval_starts: 节点的C组的区间起始位点的列表（因为macs2的peak不重叠，所以该列表就一个值）。
        :param interval_ends: 节点的C组的区间终止位点的列表（因为macs2的peak不重叠，所以该列表就一个值）。
        :param position: 位点的坐标，整数类型。
        :return:
        """
        if interval_starts[0][0] < position < interval_ends[0][0]:
            return True
        else:
            return False


if __name__ == "__main__":
    ivs = [(2, 5), (4, 7), (10, 22), (30, 87), (99, 150), ]
    it = IntervalTree()
    root = it.interval_tree(ivs)
    res = it.intervals_containing(root, 200)
    print(res)


