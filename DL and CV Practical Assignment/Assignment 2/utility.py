import numpy as np
from entity import DataArtifact,LabelsArtifact
from constant import DATA_DIR

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin-1')
    return dict
    
def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise StockException(e,sys) from e


def load_cifa100_data(data_dir):
    train_data_dic = unpickle(data_dir + "/train")
    test_data_dic = unpickle(data_dir+"/test")
    meta_data_dic = unpickle(data_dir+"/meta")
    train_data = train_data_dic["data"]
    train_fine_labels = train_data_dic["fine_labels"]
    train_coarse_labels = train_data_dic["coarse_labels"]
    test_data = test_data_dic["data"]
    test_fine_labels = test_data_dic["fine_labels"]
    test_coarse_labels = test_data_dic["coarse_labels"]
    fine_label_names = meta_data_dic['fine_label_names']
    coarse_label_names = meta_data_dic['coarse_label_names']

    train_data = train_data.reshape(len(train_data),3,32,32)
    train_data = np.moveaxis(train_data,1,3)
    train_fine_labels = np.array(train_fine_labels)
    train_coarse_labels = np.array(train_coarse_labels)

    test_data = test_data.reshape(len(test_data),3,32,32)
    test_data = np.moveaxis(test_data,1,3)
    test_fine_labels = np.array(test_fine_labels)
    test_coarse_labels = np.array(test_coarse_labels)
    data_artifact = DataArtifact(
        train_data =train_data,
        test_data = test_data
    )
    label_artifact = LabelsArtifact(
        train_fine_labels = train_fine_labels,
        test_fine_labels = test_fine_labels,
        train_coarse_labels = train_coarse_labels,
        test_coarse_labels = test_coarse_labels
    )

    return data_artifact,label_artifact