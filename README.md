# repo for various models for Neurotech club
NOTES:

  The AlterEgo attempt didn't work very well, there is another notebook I didn't upload where I stacked the channels to make a 1d input channel and it didn't work at all, so I thought I'd do a loosely based attempt thats uses conv2D. It only seems to fit the train data while the test data is just randomly trained, haven't figured out why

  # Both EEGNet attempts are based on the paper as much as I interpreted, there is probably definitely some issues
  
  The EEGNet attempt 1. The train accuracy will decrease but the test one fluctuates. It's a single participant classification and based off of some online implementations I found, I happened to chose the participant that had very low accuracy levels in comparison to the others, which is probably partly why
  
  The EEGNet attempt 2 is much more stable, but trains really slow. The notebook is a little bit misleading because I ran the train cell twice, so the result you see is after 50 epochs, not 25, which is  why it doesn't increase much during training. I haven't tuned the model much and I'll look into that, there is definitely overfitting going on. Will also look into dropping some of the channels or maybe even dropping some participants that have low accuracy when trained
  
  If you wonder why I'm not using GPU to run the model, its cause I ran out of memory and haven't yet figured out how not to run out of memory
