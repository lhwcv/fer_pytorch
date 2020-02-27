from torch.utils.data import  Dataset
from  abc import ABC,abstractmethod

class FER_Dataset(Dataset,ABC):
    def __init__(self):
        super(FER_Dataset, self).__init__()

    @abstractmethod
    def label_mapping(self):
        raise NotImplementedError()

    def build_label_mapping_fn(self, all_wanted_catgory):
        '''
        label mapping function for filter wanted data
        @param all_wanted_catgory
               e.g  [['happy'], ['sad','angry','disgust'], ['neutral']]
               means  mapping to 3 class
        '''
        label_mapping = self.label_mapping()
        map_to = {}
        for i, cs in enumerate(all_wanted_catgory):
            for c in cs:
                origin_id = label_mapping[c]
                map_to[origin_id] = i
        self.map_fn = lambda x: map_to.get(x, -1)

    def map_label_id(self, label_id_origin):
        return self.map_fn(label_id_origin)
