# vitti-proj

Implemented Neural Style Transfer (NST) using the wikiart dataset (found on wikiart.org). Neural Style Transfer is a technique in which an image's style and another's 'content' is scraped and these two are fused together into a unique image which's content is the "content" images, using the style of the "style" image.
The NST technique implemented for this project emulates Gatys et al.'s "A Neural Algorithm of Artistic Style"
For this project, the images used as "style" images in the NST were generated by a Deep Convolutional Generative Adversarial Network (DCGAN), which was trained on 3 different art genre distributions. Namely, Expressionims, Impressionism and Post-Impressionsim.
The GAN's architecture was implemented as described in Radford et al.'s 2016 paper "Unsupervised Representation Learning with Deep Convolutional
Generative Adversarial Networks"
This project was developed in Python, and the frameworks used are: Pytorch, Torchvision, Numpy and matplotlib