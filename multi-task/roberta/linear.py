import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, din=74, dout_reg=2, dout_cls=[], opt=None, **kwargs):
        super(Linear, self).__init__()

        act = bn = dropout = False

        self.linear_h = nn.Identity()
        self.linear_z = nn.Linear(din, 1)

        self.two_layer = opt and opt.get("nhidden", 0) > 0
        if opt:
            act = opt.get("act", None)
            bn = opt.get("bn", False)
            dropout = opt["dropout"]

            if int(opt.get("nhidden", 0)) > 0:
                self.linear_h = nn.Linear(din, opt["nhidden"])
                self.linear_z = nn.Linear(opt["nhidden"], 1)
                self.bn_h = nn.BatchNorm1d(opt["nhidden"]) if bn else nn.Identity()

        self.bn1 = nn.BatchNorm1d(din) if bn else nn.Identity()
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        self.bn2 = nn.BatchNorm1d(1) if bn else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        self.din = din
        self.cls_outs = []
        self.dout_reg = dout_reg

        dout = dout_reg
        for dcls in dout_cls:
            start = dout
            if dcls == 2:
                dout += 1
                stop = dout + 1

            else:
                dout += dcls
                stop = dout + dcls

            self.cls_outs.append([start, stop])

        self.linear_out = nn.Linear(1, dout)

    def get_self_dis(self, x):
        """Returns Self Disclosure binary as trained on the OMC dataset

        Args:
            x (torch.Tensor): Input Batch

        Raises:
            ValueError: If model does not have all four outputs

        Returns:
            torch.Tensor: size=(batch_size,)
        """
        with torch.no_grad():
            out = self(x)[0]
            if out.size(1) == 4:
                kraut_out = (out[:, 0] - 1) / 6
                med_out = out[:, 1] / 5
                joao_out = (out[:, 2] - 1) / 2

                omc_out = nn.Sigmoid()(out[:, 3]).cpu().numpy()
                return (kraut_out + med_out + joao_out + omc_out) / 4
            else:
                raise ValueError(
                    "This model does not have all the outputs required : found ",
                    out.size(1),
                    ", needed 4",
                )

    # define the forward pass
    def forward(self, x):
        h = self.bn1(self.dropout(x))
        if self.two_layer:
            h = self.act(self.bn_h(self.dropout(self.linear_h(h))))

        z = self.act(self.bn2(self.linear_z(h)))
        y = self.linear_out(z)

        # print(y)
        return y, z


# class Linear(nn.Module):
#     def __init__(
#         self, din=73, dout_reg=2, dout_cls=[], act=None, bn=False, dropout=0, **kwargs
#     ):
#         super(Linear, self).__init__()
#         self.trained_on = []

#         self.act = nn.ReLU() if act else nn.Identity()
#         self.bn1 = nn.BatchNorm1d(din) if bn else nn.Identity()
#         self.bn2 = nn.BatchNorm1d(1) if bn else nn.Identity()
#         self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
#         self.linear_z = nn.Linear(din, 1)
#         self.linear_reg = nn.Linear(1, dout_reg)
#         self.din = din
#         # self.cls_outs = []
#         self.dout_reg = dout_reg
#         self.linear_cls = []
#         for dout in dout_cls:
#             self.linear_cls.append(nn.Linear(1, 1 if dout == 2 else dout))

#         # dout = dout_reg
#         # for dcls in dout_cls:
#         #     start = dout
#         #     if dcls == 2:
#         #         dout += 1
#         #         stop = dout + 1

#         #     else:
#         #         dout += dcls
#         #         stop = dout + dcls

#         #     self.cls_outs.append([start, stop])

#         # self.linear2 = nn.Linear(1, dout)

#     # define the forward pass
#     def forward(self, x):
#         x = self.dropout(x)
#         x = self.bn1(x)
#         z = self.linear_z(x)
#         z = self.bn2(x)
#         z = self.act(x)
#         y_reg = self.linear_reg(z)
#         y_cls = [linear_cls(z) for linear_cls in self.linear_cls]
#         return y_reg, y_cls, z
