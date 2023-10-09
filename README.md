# Associative-rules

​											

## Introduce:

​		关联规则，其实很简单。说白了就是在一堆的数据里面，找里面某几个内容的之间是否存在着某种联系。至于是哪种联系，我们不关心。我们只关心他们之间有联系。就像，你一直暗恋的女神，突然有一天有了一个男朋友，这个时候你会悲痛欲绝。但是引起你悲痛欲绝的是你知道了你的女神有了男朋友，而至于他的男朋友是谁？其实无所谓的，不管是谁，你都会伤心。当然如果是我的话，那当然我会很开心，哈哈哈哈。其实就是开个玩笑，你的女神的男朋友不是我啊，不要来找我。

## Derivation:

### Item:

​			首先我们来介绍一下这个项（item）的概念。从它的中文来看，不是很好理解，但是看他的英文。（item） 翻译过来这个叫物品，但是不知道是哪位大师这里翻译成了项，搞得我们理解的就很迷茫。其实它这个就是物品的意思，举个例子，就比如天上的鸟，云，地下的牛，马，空气中的风，PM2.5等等。这些东西都是一些实际存在的东西，它这个东西就叫项。而我们的关联规则研究的就是这个项与项之间是不是存在着某种联系，就比如：风是一个项，马是一个项，牛也是一个项。关联规则呢，就是研究风，马，牛这三个东西之间有没有某种联系。这个就叫项与项之间的联系，既然要研究这种联系的话，那么光靠感觉是靠不住的，必须有一个度量的标准。这里我们就引出**支持度、置信度、提升度** 这三个度量标准。

### Support：

​		支持度，从这个词语表面来看，看不出这是一个什么度。我们直接来解释吧。就是**说某个项或者项集在总的数据中出现的次数和整个数据集的个数的比值**。哦，对，这里还有一个项集的概念，其实项集.项集它无非就是几个项组成的集合。就像上文的风马牛，这三个放在一起，组成一个集合{风，马，牛}，这个集合就叫项集。我们继续来解释支持度这个东西，其实就是上面的第二句话。直接来举个例子吧，看这个数据集：

![数据One](./数据One.png)

在这个数据集里面呢，我们一共有五条数据，就一行，是一个数据样本。那么现在我们算一下青绿的支持度：

$$
Support(青绿)=2/5=0.4
$$

就是青绿出现的次数，除以总的数据个数。再来看一个项集的支持度，就比如{蜷缩，清晰}：

$$
Support(蜷缩，清晰)=5/5=1
$$

最后再举一个例子，emmm...就算{乌黑，蜷缩，沉闷}这个项集的吧：

$$
Support(乌黑，蜷缩，沉闷)=1/5=0.2
$$

嗯，差不多。就是这样，这个概念很简单，和我们再贝叶斯算法里面的极大似然估计基本一样。下面我们再来看置信度这个概念。

### Confidence:

​		置信度,这个也很好理解。和我们在朴素贝叶斯算法里面讲到的条件概率是完全一样的。这个东西很简单啊。举一个简单的例子，就比如今天买一朵玫瑰花的概率是P(A),<sub>(A表示今天我买玫瑰花这一个事件)</sub>,那么在我买了一朵玫瑰花的条件下，我还会再买一朵玫瑰花嘛？可能会，也可能不会，但是，我们不得不承认，今天第一次买玫瑰花这一个事件确实对今天我第二次买玫瑰花产生了一定的影响，这很容易理解。因为我想如果没有特别特殊原因的话，我不会一天去买两朵玫瑰花，当然，除非我有两个女朋友，早上我去见了第一个女朋友，买了一束话，晚上又见了另一个女朋友，又买了一束花。好了，言归正转，现在我们要考虑我买第二朵玫瑰花发生的概率了，首先我们用这样一个符号来记录我们要得到的概率：

$$
P(B|A)
$$

很自然的，A代表的是我们今天第一次买玫瑰花的这个事件，B代表的是今天我们第二次买玫瑰花的这个事件。上面的值就代表的是，今天我在第一天买了玫瑰花的情况下，我们要第二次买到这个玫瑰花的概率。

​		换到这里，就是现在我在地里看到一个西瓜，这个西瓜被埋在了土里面，我只看到了这个西瓜的根蒂是蜷缩的，那么现在我把这个西瓜挖开，这个西瓜的纹理是清晰的概率是多少？也就是置信度是多少？很明显

![数据One](./数据One.png)

这里是：

$$
Confidence(清晰|蜷缩)=5/5=1
$$

这个我们在贝叶斯算法里面已经做了很多介绍。下面我们就来看一下提升度。

### Lift：

​		提升度，在我们的算法度量里面。应该是一个比较重要的概念了，因为**提升度代表的是项 A 的出现，对项 B 的出现概率提升的程度**。可能这样直接说，有点难以理解，我们先来看一下他的计算公式吧：

$$
Lift(A->B)=\frac {P(B|A)}{P(B)}
$$

从这个公式里面就是啥也看不出来，就是一个条件概率除以一个极大似然估计项的比值。但是不妨我们给公式的上下两边同时乘以P（A）

$$
Lift(A->B)=\frac {P(B|A)P(A)}{P(B)P(A)}
$$

根据乘法公式就等于：

$$
Lift(A->B)=\frac {P(AB)}{P(A)P(B)}
$$

同样的如果我们要求Lift(B->A)，也可以换成上面的式子：

$$
Lift(B->A)=\frac {P(AB)}{P(A)P(B)}
$$


也就是说：

$$
Lift(B->A)=Lift(A->B)
$$


这样的话，就很一目了然了。就是事件AB同时发生的概率除以事件A和事件B发生概率的乘积。众所周知，在概率论里面，如果AB两个事件是相互独立的，那么：

$$
P(AB)=P(A)P(B)
$$

在这个时候Lift(A->B)的值就等于1。说明他们两个互不影响，谁多谁少都无所谓的。但是如果，他们的值不等于1呢？这个时候就说明A和B之间是存在这某种联系的。那这种联系无非就是两种情况。第一种就是A多了，B也多了。另一种，A多了，B少了。无非就这两种可能性。这个时候，我们提升度不就是正好可以划分这两种情况吗？想象一下，假设我们现在有一个数据集D，它里面16个数据。其中A出现了8次，B出现了8次。当然A和B属于不同的项。那么P(A)=0.5，P(B)=0.5当P（AB）=0.25时，也就是A和B同时出现了2次时他们相互独立，当P（AB）>0.25时，A和B同时出现了2次以上，Lift就是大于1的，当P（AB）<0.25时，也就是A和B同时出现了一次或者一次没有，Lift就是小于1的。从这个例子里面可以看出来，Lift大于1，代表着A出现的越多B出现的越多，当Lift<1,代表着A出现的越多，B出现的越少。反过来也是一样的，因为我们前面证明了AB和BA的Lift是一样的。我们将它表示为分段函数：

$$
Crate(A,B)=\begin{cases} 
		A和B正相关, & lift>1\\ 
		A和B不相关, & lift=1 \\
		A和B负相关, & lift<1		
\end{cases}
$$

### Frequent Item Sets：

​		频繁项集，其实吧就是相对来说出现的次数多的项集。那么出现多少次算多呢？这个不好说，我们得有一个度量的标准。在这里我们是用支持度来衡量的，当支持度大于等于某一个值时，这个集合就叫频繁项集，相反的就是不平凡项集。这里我们假设这个阈值为α。那么这个就能表示为这样：

$$
Set(A)=\begin{cases} 
		A为频繁项集合, & Support>=α\\ 
		A为非频繁项集, & Support<α \\		
\end{cases}
$$

### Apriori:

​	介绍完了上面的相关概念，我们来看一下这个apriori算法的一个思想。其实吧，它很简单。就是**频繁项集的子集都是频繁项集合*****非平凡项集的超集，都是非平凡项集合**，就这两条。这实在是太容易理解了。就好比是，一群玫瑰花里面，随便摘一朵，都是玫瑰花。而一个羊粪蛋，撒上再多的香料也还是羊粪蛋。就是这个样子的，但是这样简单的两条，它也确实有很大的作用，可以优化很多计算量。为了文章的美观，我们这里放上一张关于这个算法的图片，图片是网上找的，哈哈哈。

![img2](D./img2.png)

## Code：

### Data：

![data2](./data2.png)

当然这里只是展示了一部分，具体可以到[UCL](http://archive.ics.uci.edu/ml/index.php)这个网站下载。

### Tree:

![img3](D./img3.png)

### Result:

![result](./result.png)

截取了其中的一部分。

### Statement：

​			**_由于近期的作业和事情实在太多，所以代码是找网上抄的，其它部分均为原创。_**

### Code：

```python
def loadData(path=''):
    '''
    加载原始数据
    '''
    f = open(path + 'house-votes-84.csv')
    txt = f.read()
    f.close()
    lst_txt = txt.split('\n')
    data = []
    for txt_line in lst_txt:
        tmp = txt_line.split(',')
        data.append(tmp)
    return data
def preProcessing(data, vote_y_n):
    '''
    数据预处理
    以按 y 或 n 寻找关联规则
    '''
    data_pre = []
    for data_line in data:
        tmp_data = []
        for i in range(1, len(data_line)):
            # 从第二列开始，将数据文件中的记录与当前的选择vote_y_n进行比较，若找到了相关记录，把下标存进去
            if (data_line[i] == vote_y_n):
                tmp_data.append(i)
        if (tmp_data == []):
            continue  # 如果当前这一条记录中没有任何一个项是vote_y_n对应的选项，那么不存储空列表，直接进行下一个记录的查找
        data_pre.append(tmp_data)
    return data_pre

def ppreProcessing(data, vote_y_n, party):
    '''
    数据预处理
    以按 y 或 n 和议员所属党派来寻找关联规则
    '''
    data_pre = []
    for data_line in data:
        tmp_data = []
        if data_line[0] == party:
            for i in range(1, len(data_line)):
                # 从第二列开始，将数据文件中的记录与当前的选择vote_y_n进行比较，若找到了相关记录，把下标存进去
                if (data_line[i] == vote_y_n):
                    tmp_data.append(i)
        if (tmp_data == []):
            continue  # 如果当前这一条记录中没有任何一个项是vote_y_n或者这条记录不是对应party的议员对应的选项，那么不存储空列表，直接进行下一个记录的查找
        data_pre.append(tmp_data)
    return data_pre

def rule_mining(data, support, confidence):
    '''
    挖掘关联规则
    '''
    dic_1 = mining_first(data, support, confidence)
    # print(dic_1)
    dic_2 = mining_second(data, dic_1, support, confidence)
    # print(dic_2)
    dic_before = dic_2

    dic_r = []

    # 频繁项集产生的终止条件就是不再有新的频繁项集产生为止
    while (dic_before != {}):
        # dict_r里面存储的是频繁2-项集及之后的所有频繁项集
        dic_r.append(dic_before)
        dic_3 = mining_third(data, dic_before, support, confidence)
        dic_before = dic_3
    return dic_r
    pass

def mining_first(data, support, confidence):
    '''
    进行第一次挖掘
    挖掘候选1-项集
    '''
    dic = {}
    count = len(data)
    for data_line in data:
        # 对于数据集中的每一行投票数据
        for data_item in data_line:
            # 对于每一行数据中的下标（对应某个议题）
            if (data_item in dic):
                # 以键值对的形式进行存储和计数
                dic[data_item] += 1
            else:
                dic[data_item] = 1

    assert (support >= 0) and (support <= 1), 'suport must be in 0-1'
    # 依靠给定的支持度阈值和投票数据的总数的得到满足条件的最小支持度值
    val_suport = int(count * support)
    assert (confidence >= 0) and (confidence <= 1), 'coincidence must be in 0-1'
    # 如果键值对中的值大于或等于当前支持度阈值，则可以将该键值对作为频繁1-项集保留
    dic_1 = {}
    for item in dic:  # 如果对每一个议题的所选定的（y|n）进行计数，若计数总值超过了支持度所需要的计数，就把它放到下一个字典中
        if (dic[item] >= val_suport):
            dic_1[item] = dic[item]

    return dic_1

def mining_second(data, dic_before, support, confidence):
    '''
    进行关联规则的二次挖掘
    挖掘出候选2-项集

    注：所有挖掘出来的频繁项集都是以字典的形式存储的，字典的键是频繁项集，
    1频繁项集用1-16个整数，表示这些议题在原数据集中的下标；多频繁集就是这些下标的一个元组
    隐藏含义是这些议题共同被投票为vote_y_n，字典的值就是这样的组合出现的次数
    '''
    # 每一次扩展频繁项集的时候产生一个临时dict用于保存那些通过频繁项集生成算法可以留下的项集
    # 但是还要对其中的结果进行支持度判断，才能确定最终留下的算法
    dic = {}
    count = len(data)
    count2 = 0
    for data_line in data:
        # 获取元素数量
        count_item = len(data_line)
        # 每两个组合计数
        for i in range(0, count_item - 1):
            # 外层循环，控制频繁2-项集中的第一个元素的取值
            for j in range(i + 1, count_item):
                # 内层循环，控制频繁2-项集中的第二个元素的取值
                if (data_line[i] in dic_before and data_line[j] in dic_before):

                    count2 += 1
                    tmp = (data_line[i], data_line[j])
                    if (tmp in dic):
                        # 上同，使用键值对集合计数，只不过此时元素是二元的元组
                        dic[tmp] += 1
                    else:
                        dic[tmp] = 1
                else:
                    continue
                    # 当两个项中有一个不是频繁1-项集，根据剪枝策略，这样组成的项不是频繁2-项集
    # print(dic)
    assert (support >= 0) and (support <= 1), 'suport must be in 0-1'
    assert (confidence >= 0) and (confidence <= 1), 'confidence must be in 0-1'

    dic_2 = {}
    for item in dic:
        count_item0 = dic_before[item[0]]
        count_item1 = dic_before[item[1]]
        # 判断 支持度 和 置信度
        # 判断置信度的时候对于一个无序的元组，任何一种方向的规则都有可能，都要进行比较
        if ((dic[item] >= support * count) and (
                (dic[item] >= confidence * count_item0) or (dic[item] >= confidence * count_item1))):
            dic_2[item] = dic[item]

    return dic_2


def mining_third(data, dic_before, support, confidence):
    '''
    进行关联规则的三次挖掘
    挖掘出候选3-项集或者4-项集乃至n-项集
    '''
    # 频繁项集的产生使用Fk-1*Fk-1的策略
    dic_3 = {}
    for item0 in dic_before:
        # 外层循环控制频繁k-1项集中的某一项
        for item1 in dic_before:
            # 内层循环控制频繁k-1项集中的另一项
            if (item0 != item1):
                # print(item0,item1)
                item_len = len(item0)
                equal = True
                tmp_item3 = []
                # 判断前n-1项是否一致
                for i in range(item_len - 1):
                    tmp_item3.append(item0[i])
                    if (item0[i] != item1[i]):
                        equal = False
                        break
                if (equal == True):
                    # 如果两个Fk-1项具有k-2个公共前缀，那么就按照顺序，将其组合起来
                    minitem = min(item0[-1], item1[-1])
                    maxitem = max(item0[-1], item1[-1])
                    tmp_item3.append(minitem)
                    tmp_item3.append(maxitem)
                    tmp_item3 = tuple(tmp_item3)
                    dic_3[tmp_item3] = 0
                else:
                    continue
    # print('dic_3:',dic_3)
    # 暴力统计支持度的方法，对于每一个数据项，看每个新找到的k项集是否包含在数据项中
    # 比较的方法，是对项的每一位进行判断，看这一位是否在数据项中

    for data_line in data:
        for item in dic_3:
            is_in = True
            for i in range(len(item)):
                if (item[i] not in data_line):
                    is_in = False

            # 该候选k项集中的所有项都在数据项中，则可以将该项保留
            if (is_in == True):
                dic_3[item] += 1

    assert (support >= 0) and (support <= 1), 'suport must be in 0-1'
    assert (confidence >= 0) and (confidence <= 1), 'coincidence must be in 0-1'

    count = len(data)
    dic_3n = {}
    for item in dic_3:
        # 前一项的支持度计数，就是现在的项除去末尾的数字，通过键值对在原来的字典中查询的值
        count_item0 = dic_before[item[:-1]]
        # 判断 支持度 和 置信度
        if ((dic_3[item] >= support * count) and (dic_3[item] >= confidence * count_item0)):
            dic_3n[item] = dic_3[item]

    return dic_3n


def association_rules(freq_sets, min_conf):
    '''
    根据产生的频繁项集生成满足置信度要求的规则

    :param dict: 频繁项集的字典
    :param dict: 频繁项集字典中的频繁项集列表
    :param min_conf: 最小置信度
    :return: 规则列表
    '''

    rules = []
    max_len = len(freq_sets)

    for k in range(max_len - 1):
        for freq_set in freq_sets[k]:
            for sub_set in freq_sets[k + 1]:
                if set(freq_set).issubset(set(sub_set)):
                    conf = freq_sets[k + 1][sub_set] / freq_sets[k][freq_set]
                    rule = (set(freq_set), set(sub_set) - set(freq_set), conf)
                    if conf >= min_conf:
                        rules.append(rule)
    return rules


if (__name__ == '__main__'):
    data_row = loadData()

    data_y = preProcessing(data_row, 'y')
    data_n = preProcessing(data_row, 'n')
    data_y_republican = ppreProcessing(data_row, 'y', 'republican')
    data_y_democrat = ppreProcessing(data_row, 'y', 'democrat')
    data_n_republican = ppreProcessing(data_row, 'n', 'republican')
    data_n_democrat = ppreProcessing(data_row, 'n', 'democrat')

    # 支持度
    support = 0.3
    # 置信度
    confidence = 0.9

    # 总的y规则与两个党派的y规则
    r_y = rule_mining(data_y, support, confidence)
    print('vote `y`:\n', r_y)
    rule_y = association_rules(r_y, confidence)
    print('rule `y`:\n', rule_y)

    r_y_republican = rule_mining(data_y_republican, support, confidence)
    print('vote_republican `y`:\n', r_y_republican)
    rule_y_republican = association_rules(r_y_republican, confidence)
    print('rule_republican `y`:\n', rule_y_republican)

    r_y_democrat = rule_mining(data_y_democrat, support, confidence)
    print('vote_democrat `y`:\n', r_y_democrat)
    rule_y_democrat = association_rules(r_y_democrat, confidence)
    print('rule_democrat `y`:\n', rule_y_democrat)

    # 总的n规则与两个党派的n规则
    r_n = rule_mining(data_n, support, confidence)
    print('vote `n`:\n', r_n)
    rule_n = association_rules(r_n, confidence)
    print('rule `n`:\n', rule_n)

    r_n_republican = rule_mining(data_n_republican, support, confidence)
    print('vote_republican `n`:\n', r_n_republican)
    rule_n_republican = association_rules(r_n_republican, confidence)
    print('rule `n`:\n', rule_n_republican)

    r_n_democrat = rule_mining(data_n_democrat, support, confidence)
    print('vote_democrat `n`:\n', r_n_democrat)
    rule_n_democrat = association_rules(r_n_democrat, confidence)
    print('rule_democrat `n`:\n', rule_n_democrat)

    f = open('result_mining.txt', 'w')
    f.write('vote `y`:\n')
    f.write(str(r_y))
    f.write('rule `y`:\n')
    f.write(str(rule_y))

    f.write('\n\nvote `n`:\n')
    f.write(str(r_n))
    f.write('rule `n`:\n')
    f.write(str(rule_y))
    f.close()
```
