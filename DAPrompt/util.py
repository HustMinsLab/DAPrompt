import torch
def isSort(prediction,label,sort_rate):
    prediction=torch.softmax(prediction,0)
    prediction=prediction.detach().cpu().tolist()
    temp=prediction[label]
    if temp > sort_rate:
        return True
    return False


def convert(prediction,batch_event):
    to_save_prediction = torch.softmax(prediction, dim=1)
    to_save_prediction = to_save_prediction.detach().cpu().tolist()
    event_predt = []
    event_place = []
    nothing_predt = []
    nothing_place = []
    for temp_i in range(len(prediction)):
        event_predt.append(to_save_prediction[temp_i][batch_event])
        nothing_predt.append(to_save_prediction[temp_i][50270])
        to_save_prediction[temp_i].sort(reverse=True)
        to_save_prediction[temp_i] = to_save_prediction[temp_i][:10]

    for temp_i in range(len(prediction)):
        if event_predt[temp_i] in to_save_prediction[temp_i]:
            event_place.append(to_save_prediction[temp_i].index(event_predt[temp_i]))
        else:
            event_place.append(-1)
        if nothing_predt[temp_i] in to_save_prediction[temp_i]:
            nothing_place.append(to_save_prediction[temp_i].index(nothing_predt[temp_i]))
        else:
            nothing_place.append(-1)
    return to_save_prediction,event_place,nothing_place


