# Face-Anti-Spoofing
In this project, we have re-trained the ["Learning Generalized Spoof Cues for Face Anti-spoofing"](https://arxiv.org/abs/2005.03922) paper from scratch with Oulu (protocol 1) dataset. In order to balance the real and fake data, we extracted 15 frames from each subject’s fake videos and 60 frames from that subject’s real video. Therefore, we obtained 15084 frames with the help of the facenet-pytorch algorithm, which contains 7380 pristine frames and 7704 manipulated frames. (Note that the missed actual frames were due to the disability of the face recognition algorithm to detect the person’s face.)

## Requirement
Considering that this project was implemented based on the PaddlePaddle platform, please install the paddle 1.7.1 version.
```
python -m pip install paddlepaddle-gpu==1.7.1.post107 
```


## Results
In the "cue" subfolder, you can find the output of the U-net for a variety of subjects. We achieved 94.17% accuracy for validation and 91.43% for test data. The AUC value was calculated as 98.85%, and the best threshold was 0.7501.
In what follows, we present two samples of real videos and three examples of replayed attacks. It should be mentioned that to evaluate the robustness of the trained network, we tested under a variety of situations such as daylight with both laptop and cellphone cameras. Given that the naturality of the Oulu dataset does not contain samples recorded in darkness, the trained network sometimes faces inaccuracy in the situation with a lack of lighting.


+ Real Videos\
\
![Face-Anti-Spoofing](SampleOutput/real.gif)

+ Replay attack\
\
![Face-Anti-Spoofing](SampleOutput/fake.gif)
\
## Announcement
Please feel free to contact me if you have any questions or you want the pre-trained model. 
