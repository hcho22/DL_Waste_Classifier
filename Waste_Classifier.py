import cv2
import fastai
from fastai import *
from fastai.vision import *
import pathlib

path = pathlib.PosixPath('data/Kaggle Garbage Data/')
data = ImageDataBunch.from_folder(path=path, ds_tfms=get_transforms(), size=224, valid_pct=0.2)
data.normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.load('waste-kaggle-stage-1');

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()   #capture each frame
    cv2.imshow('FRAME',frame)
    
    #For capturing frame and saving it as an image at given folder:
    #if cv2.waitKey(1) == ord('n'):
    cv2.imwrite('test1.jpg',frame)
    img = open_image(pathlib.PosixPath('test1.jpg'))
        
    print(learn.predict(img))

    # Prediction of Waste Type
    label,index, pred = learn.predict(img)
    cv2.putText(frame, "Waste Tye = "+ str(label), (380, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 215), 2)
    cv2.putText(frame, "Prob = {0:.4f}".format(torch.max(pred).item()), (380, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("FRAME",frame)
    print("Current Waste Type =\n",torch.max(pred).item())
    print("\t\t "+str(label)+"\t\t")

    #For quitting the given session: ord is used to obtain unicode of the given string.
    #cv2.waitKey returns the unicode of the key which is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
