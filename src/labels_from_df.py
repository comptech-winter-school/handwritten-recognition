import os
import pandas as pd

df = pd.read_csv("/content/assignments_from_pool_601263__05-12-2020.tsv", delimiter='\t' )

labels_dict = {}
for image_name in os.listdir("/content/werner"):
  label = df.loc[df["INPUT:image"] == "".join(("/HANDWRITTEN/werner_1000_part_1_181120/", image_name)), "OUTPUT:output"]
  label = label.tolist()
  if label != []: 
    labels_dict.update({image_name:label[0]})
labels_dict