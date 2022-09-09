

from constant import DATA_DIR, MODEL_DIR
from utility import load_cifa100_data

# data_artifact:DataArtifact
# label_artifact:LabelsArtifact

data_artifact,label_artifact=load_cifa100_data(DATA_DIR)
train_data = data_artifact.train_data.astype('float32')/255.0
test_data =data_artifact.test_data.astype('float32')/255.0

from keras.models import load_model

model = load_model(filepath=MODEL_DIR)
print(model.summary())
scores = model.evaluate(train_data,label_artifact.train_fine_labels)
print(f"Accuracy {scores[1]*100}")
