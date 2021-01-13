from django.shortcuts import render
from django.http import JsonResponse,HttpResponse

#from rest_framework.views import APIView
#from rest_framework.response import Response
import os
from django.contrib import messages
from .NER import NERClass


# Create your views here.
ner = NERClass()
dirname = os.path.dirname(os.path.abspath(__file__))
def index(request):
    if not os.path.isfile(dirname+'/metrics.txt'):
            return render(request = request,
                  template_name = 'main/index.html',
                  context = {})

    else:
        steps, trainloss, evalLoss, evalAccuracy = ReadMetrics()
        train_size, val_size, vocab_size = len(ner.vocab), len(ner.t_sentences), len(ner.v_sentences)
        #train_size, val_size, vocab_size =  100, 100, 100
        return render(request = request,
                template_name = 'main/index.html',
                context = {"steps": steps, "trainloss":trainloss,"evalLoss":evalLoss,"evalAccuracy":evalAccuracy,
                            "train_size":train_size, "val_size":val_size,"vocab_size":vocab_size })

    return render(request = request,
                    template_name = "main/index.html",
                    context= {})


def ReadMetrics():
    steps, trainloss, evalLoss, evalAccuracy = [], [], [], []
    with open(dirname+'/metrics.txt') as f:
        i = 0
        for line in f:
            line = line.strip().replace(':','').split()
            if '|' in line:
                if i == 0:
                    steps.append(int(line[1]))
                    trainloss.append(float(line[line.index('|')+1]))
                elif i == 1:
                    evalLoss.append(float(line[line.index('|')+1]))
                elif i == 2:
                    evalAccuracy.append(float(line[line.index('|')+1]))
                    i = -1
    
                i += 1
    return steps, trainloss, evalLoss, evalAccuracy

def readFinishConfig():
    train_size = val_size = vocab_size = 0
    with open(dirname+'/size.txt','r') as f:
        for line in f:
            line = line.strip().split('=')
            if line[0] == 'vocab_size':
                vocab_size = float(line[1])
            elif line[0] == 'train_size':
                train_size = float(line[1])
            elif line[0] == 'val_size':
                val_size = float(line[1])
    return train_size, val_size, vocab_size



def train(request):
    if request.method == "POST":
        steps = int(request.POST['steps'])

        with open(dirname+"/trainingconfig.txt",'w') as f:
            f.write('steps='+str(steps)+'\n')

        vocab_size, train_size, val_size = ner.trainingModel(train_steps= steps)
        #vocab_size, train_size, val_size =  100,100,100 #For debug

        
        if not os.path.isfile(dirname+'/metrics.txt') :
            print("Error not generate metrics")
            messages.error(request, "Metrics file not generated. Error for training")
        else:
            steps, trainloss, evalLoss, evalAccuracy = ReadMetrics()

            data = {
            "steps":steps,
            "trainloss": trainloss,
            "evalLoss": evalLoss,
            "evalAccuracy":evalAccuracy,
            "train_size": train_size,
            "val_size": val_size,
            "vocab_size":vocab_size
            }
            return JsonResponse(data)
    return HttpResponse('')

def evaluation(request):
    if request.method == "POST":
        sentence = request.POST['sentence'].strip()
       
        preds, labels = ner.predict(sentence=sentence)
        #preds, labels = ["B-geo","B-tim"],  ["B-Geographical Entity","B-Time Indicator"]   #For debug
        words, entities = [], []
        for w, e in zip(sentence.split(),labels):
            if e != 'filler word':
                words.append(w)
                entities.append(e)
        data = {
            "words":words,
            "entities": entities
        }

        return JsonResponse(data)
    return HttpResponse('')