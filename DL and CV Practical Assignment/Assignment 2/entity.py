from collections import namedtuple


DataArtifact = namedtuple('DataArtifact',['train_data','test_data'])
LabelsArtifact = namedtuple('LabelArtifact',['train_fine_labels','test_fine_labels','train_coarse_labels','test_coarse_labels'] )