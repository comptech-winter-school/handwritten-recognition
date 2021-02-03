from torchvision import models
from torch import nn



class ResNetOCR(nn.Module):
  def __init__(self, n_classes, hidden_rnn):
    super(ResNetOCR,self).__init__()
    model = nn.Sequential(*list(resnet18(pretrained=False).children())[:-4])
    self.fc1 = nn.Linear(1024, hidden_rnn)
    self.cnn = model
    self.rnn = nn.GRU(hidden_rnn, 32 , bidirectional=True,
                          num_layers=2,
                          batch_first=True)
    self.output = nn.Linear(64,n_classes+1)
  
  def forward(self, images, labels=None, len_labels = None):
    bs, c, h, w = images.size()
    x = self.cnn(images)
    #print(x.size())
    x = x.permute(0,3,1,2)
    #print(x.size())
    x = x.view(bs, x.size(1),-1)
    #print(x.size()[2])
    x = self.fc1(x)
    x, _ = self.rnn(x)
    x = self.output(x)
    #print(x.size())
    # permute again 
    x = x.permute(1,0,2)
    #print(x)
    if labels is not None:
      log_softmax_values =  F.log_softmax(x,2)   
      input_lenghts = torch.full(size=(bs,),
                                 fill_value=log_softmax_values.size(0), 
                                 dtype = torch.int32
                                 )
      #print(input_lenghts)
         
      loss = nn.CTCLoss(blank=0,zero_infinity = True)(
          log_softmax_values,
          labels,
          input_lenghts,
          len_labels
      )  
      
      return x, loss
    return x



class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2, 2]
        ps = [1, 1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        cnn.add_module('pooling{0}'.format(4),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        #convRelu(7, True)  # 512x1x16


        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))


    def forward(self, images, labels=None, len_labels = None):
        # conv features
        conv = self.cnn(images)
        b, c, h, w = conv.size()
        #assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        #print(conv.size())
        print(conv.size())
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        
        # add log_softmax to converge output
        x = output
        print(x.size())
        if labels is not None:
          log_softmax_values =  F.log_softmax(x,2)   
          input_lenghts = torch.full(size=(b,),
                                     fill_value=log_softmax_values.size(0), 
                                     dtype = torch.int32
                                     )
          loss = nn.CTCLoss(blank=0,zero_infinity = True)(
              log_softmax_values,
              labels,
              input_lenghts,
              len_labels
          )  
          
          return x, loss

        return x


class OcrModel_mobilenet(nn.Module):
    def __init__(self, num_characters):
        super(OcrModel_mobilenet, self).__init__()
        mobilenet = models.mobilenet_v2()
        self.mobilenet_feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
        self.linear1 = nn.Linear(int(20480/64), 64)
        self.dropout1  = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32 , bidirectional=True,
                          num_layers=2,
                          dropout=0.25,
                          batch_first=True)
        self.output = nn.Linear(64, num_characters + 1)
        
    def forward(self, images, labels=None, len_labels=None):
        bs, c, h, w = images.size()
        out = F.relu(self.mobilenet_feature_extractor(images))# 32 1280 7 7
        #print(out.size())

        x = out.permute(0,3,1,2) # 32 10, 1280, 3
        #print(x.size())
        x = torch.reshape(x,(bs, 64, int(20480/64)))#32 49 1280
        #print(x.size())
        x = self.linear1(x)
        x = self.dropout1(x)
        #print(x.size())


        x, _ = self.gru(x)
        #print(x.size())
        x = self.output(x)
        #print(x.size())
        # permute again 
        x = x.permute(1,0,2)
        #print(x.size())
        if labels is not None: 
          log_softmax_values =  F.log_softmax(x,2)   
          input_lenghts = torch.full(size=(bs,),
                                     fill_value=log_softmax_values.size(0), 
                                     dtype = torch.int32
                                     )
          #print(input_lenghts)
          
          loss = nn.CTCLoss(blank=0,zero_infinity = True)(
              log_softmax_values,
              labels,
              input_lenghts,
              len_labels
          )  
          
          return x, loss
        return x


class OcrModel_vgg16(nn.Module):
    def __init__(self, num_characters):
        super(OcrModel_vgg16, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        self.vgg_feature_extractor = nn.Sequential(*list(vgg.children())[:-1])
        self.linear1 = nn.Linear(256, 64)
        self.dropout1  = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32 , bidirectional=True,
                          num_layers=2,
                          dropout=0.25,
                          batch_first=True)
        self.output = nn.Linear(64, num_characters + 1)
        
    def forward(self, images, labels=None, len_labels=None):
        bs, c, h, w = images.size()
        out = F.relu(self.vgg_feature_extractor(images))
        #print(out.size())

        x = out.permute(0,3,1,2) # 32 7 512  7 
        #print(x.size()) #512 7 7 
        x = torch.reshape(x,(bs, 7*7*2, 256))#view(bs, 7*7*2, 256)  # 32 7 512*7 
        #print(x.size()[2])
        x = self.linear1(x)
        x = self.dropout1(x)
        #print(x.size())


        x, _ = self.gru(x)
        #print(x.size())
        x = self.output(x)
        #print(x.size())
        # permute again 
        x = x.permute(1,0,2)
        #print(x.size())
        if labels is not None: 
          log_softmax_values =  F.log_softmax(x,2)   
          input_lenghts = torch.full(size=(bs,),
                                     fill_value=log_softmax_values.size(0), 
                                     dtype = torch.int32
                                     )
          
          loss = nn.CTCLoss(blank=0, zero_infinity = True)(
              log_softmax_values,
              labels,
              input_lenghts,
              len_labels)
          
          return x, loss
        return x