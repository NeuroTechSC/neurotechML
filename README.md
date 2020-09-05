# repo for various models for Neurotech club
NOTES:

  The AlterEgo attempt is loosely based off of alter ego. The train accuracy will decrease but the test one fluctuates wildly. It's a single participant classification and based off of some online implementations I found, I happened to chose the participant that had very low accuracy levels in comparison to the others, which is probably why
  
  The EEGNet attempt is much more stable, the notebook is a little bit misleading because I ran the train cell twice, so the result you see is after 50 epochs, not 25, which is  why it doesn't increase much during training. I haven't tuned the model much and I'll look into that, there is definitely overfitting going on. Will also look into dropping some of the channels or maybe even dropping some participants that have low accuracy when trained
  
  If you wonder why I'm not using GPU to run the model, its cause I ran out of memory and haven't yet figured out how not to run out of memory
