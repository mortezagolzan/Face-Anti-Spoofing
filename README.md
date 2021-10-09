# Face-Anti-Spoofing
In this project, we have re-trained the ["Learning Generalized Spoof Cues for Face Anti-spoofing"](https://arxiv.org/abs/2005.03922) paper from scratch with Oulu (protocol 1) dataset. In order to balance the real and fake data, we extracted 15 frames from each subject’s fake videos and 60 frames from that subject’s real video. Therefore, we obtained 15084 frames with the help of the facenet-pytorch algorithm, which contains 7380 pristine frames and 7704 manipulated frames. (Note that the missed actual frames were due to the disability of the face recognition algorithm to detect the person’s face.)

## Requirement
Considering that this project was implemented based on the paddlepaddle platform, please install the paddle 1.7.1 version.
```
python -m pip install paddlepaddle-gpu==1.7.1.post107 
```


## Results
![Face-Anti-Spoofing](SampleOutput/real.gif)
![Face-Anti-Spoofing](SampleOutput/fake.gif)
