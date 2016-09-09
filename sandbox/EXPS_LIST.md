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
      - model meta parameters are all fixed to default.
      - learning rule : Adam, Clip
  - Result :
