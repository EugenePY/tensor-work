Experiments
===
To test partial information attention mechenisim can learn a meaningful policy.

0. Mnist dataset


1. Transform Mnist dataset
  - test the attention can detect the handwrite digits
    - experiment configs:
      - image size : (60, 60) with mnist (28, 28) inside
      - glimpse type : 
        - simple glimpse
        - size : (15, 15) | enough to cover the digits
        - stride : (15, 15)
      - n_step : 15
      - random initial the firt loction state
      - model meta parameters are all fixed to default.
      - learning rule : Adam, Clip

  - Result :
  - Observation :
    - The policy trend to start at some fix location then decide the next step.
    - The REINFORCE algorithm seems not suitable for this kind of task. It is because the simple glimpse do not provide enough side information (pooling a wider range of image). It seens be better to have a pretrain step to tell the network a baseline policy. This observation can be seen from the gradient of the REINFORCE algorithm. It is because the model not only need to learn how to glimpse but also need to learn how to reconize the image from the glimpse. Maybe we can pretrain the model and do not update the weights of location network. This dynamic of model may not be converge in a given update. More details pls refer to the sandbox/docs/dynamics.pdf.
