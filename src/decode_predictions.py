import torch




def decode_preds(preds, encoder):
    preds = preds.permute(1,0,2)
    preds = torch.softmax(preds,2)
    preds = torch.argmax(preds,2)
    preds = preds.detach().cpu().numpy()
    preds_list  = []
    for i in range(preds.shape[0]):
        tmp = []
        for j in preds[i,:]:
            j = j-1
            if j == -1:
                tmp.append("*")
            else:
                tmp.append(encoder.inverse_transform([j])[0])
        element = "".join(tmp)
        preds_list.append(element)
    return preds_list
    