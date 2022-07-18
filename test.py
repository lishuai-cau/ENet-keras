#---------------------------------------------#
#   该部分用于查看网络结构
#---------------------------------------------#
from nets.ENet import enet
if __name__ == "__main__":
    model = enet(2, input_height=224, input_width=224)
    model.summary()

