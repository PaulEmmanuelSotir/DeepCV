"""
# TODO: Video interpolation / dynamics learning and also relevant for unsupervised keypoint detection ideas: https://github.com/google-research/google-research/tree/master/video_structure from this paper: https://arxiv.org/abs/1906.07889 
# TODO: NN model for Keypoints proposal from a conv NN which outputs K feature maps: each output channel is normalized and averaged into (x,y) coordinates in order to obtain relevant keypoints (K at most). Trained using a autoencoder setup: a generator (decoder with end-to-end skip connection from anchor frame) must be able to reconstruct input image from keypoints (converted to gaussian heat maps) and another frame along with its own keypoints (e.g. first video frame)
# TODO: Modify keypoint model in order to have feature pattern information associated with keypoint coordinates (instead of simply associate input image)?
"""
