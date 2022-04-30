The folder contains three files:

1. extracted_features.csv: the x64 features extracted from the original raw images using a CNN.
The data contains a total of 2220 samples.

2. labels.csv contains the label mapping for each of the input samples. 
Labels are mapped as follows:

{0:'Pierre-Auguste_Renoir',
 1:'Raphael',
 2:'Leonardo_da_Vinci',
 3:'Sandro_Botticelli',
 4:'Francisco_Goya',
 5:'Vincent_van_Gogh',
 6:'Pablo_Picasso',
 7:'Albrecht_Durer',
 8:'Others'}

3.raw_images.csv contains the scaled pixel intensities of the raw input images. Each raw contains: 150x150x3 values
In ordor to visualize a given image img[i], you can run:
  plt.imshow(img[i].reshape(150,150,3))