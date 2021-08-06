from config.config import cfg
from collections import OrderedDict

LABELS = {}
LABELS['Cityscapes'] = {
            'unlabeled'            :  0 ,  
            'ego vehicle'          :  1 ,  
            'rectification border' :  2 ,  
            'out of roi'           :  3 ,  
            'static'               :  4 ,  
            'dynamic'              :  5 ,  
            'ground'               :  6 ,  
            'road'                 :  7 ,  
            'sidewalk'             :  8 ,  
            'parking'              :  9 ,  
            'rail track'           : 10 ,  
            'building'             : 11 ,  
            'wall'                 : 12 ,  
            'fence'                : 13 ,  
            'guard rail'           : 14 ,  
            'bridge'               : 15 ,  
            'tunnel'               : 16 ,  
            'pole'                 : 17 ,  
            'polegroup'            : 18 ,  
            'traffic light'        : 19 ,  
            'traffic sign'         : 20 ,  
            'vegetation'           : 21 ,  
            'terrain'              : 22 ,  
            'sky'                  : 23 ,  
            'pedestrian'           : 24 ,  # originally named as "person"
            'rider'                : 25 ,  
            'car'                  : 26 ,  
            'truck'                : 27 ,  
            'bus'                  : 28 ,  
            'caravan'              : 29 ,  
            'trailer'              : 30 ,  
            'train'                : 31 ,  
            'motorcycle'           : 32 ,  
            'bicycle'              : 33 ,  
            'license plate'        : -1 ,  
}

LABELS['SYNTHIA'] = {
            'void'          : 0 , 
            'sky'           : 1 ,
            'building'      : 2 ,
            'road'          : 3 ,
            'sidewalk'      : 4 ,
            'fence'         : 5 ,
            'vegetation'    : 6 ,
            'pole'          : 7 ,
            'car'           : 8 ,
            'traffic sign'  : 9 ,
            'pedestrian'    : 10,
            'bicycle'       : 11,
            'motorcycle'    : 12,
            'parking-slot'  : 13,
            'road-work'     : 14,
            'traffic light' : 15,
            'terrain'       : 16,
            'rider'         : 17,
            'truck'         : 18,
            'bus'           : 19,
            'train'         : 20,
            'wall'          : 21,
            'lanemarking'   : 22,
}

LABELS['GTAV'] = {
            'unlabeled'            :  0 ,  
            'ego vehicle'          :  1 ,  
            'rectification border' :  2 ,  
            'out of roi'           :  3 ,  
            'static'               :  4 ,  
            'dynamic'              :  5 ,  
            'ground'               :  6 ,  
            'road'                 :  7 ,  
            'sidewalk'             :  8 ,  
            'parking'              :  9 ,  
            'rail track'           : 10 ,  
            'building'             : 11 ,  
            'wall'                 : 12 ,  
            'fence'                : 13 ,  
            'guard rail'           : 14 ,  
            'bridge'               : 15 ,  
            'tunnel'               : 16 ,  
            'pole'                 : 17 ,  
            'polegroup'            : 18 ,  
            'traffic light'        : 19 ,  
            'traffic sign'         : 20 ,  
            'vegetation'           : 21 ,  
            'terrain'              : 22 ,  
            'sky'                  : 23 ,  
            'pedestrian'           : 24 ,  # originally named as "person"
            'rider'                : 25 ,  
            'car'                  : 26 ,  
            'truck'                : 27 ,  
            'bus'                  : 28 ,  
            'caravan'              : 29 ,  
            'trailer'              : 30 ,  
            'train'                : 31 ,  
            'motorcycle'           : 32 ,  
            'bicycle'              : 33 ,  
            'license plate'        : 34 ,  
}

LABEL_TASK = {}
LABEL_TASK['SYNTHIA2Cityscapes'] = OrderedDict({'sky': 0, 'building': 1, 'road': 2, 'sidewalk': 3, 'fence': 4, 'vegetation': 5, 
             'pole': 6, 'car': 7, 'traffic sign': 8, 'pedestrian': 9, 'bicycle': 10, 'motorcycle': 11, 
             'traffic light': 12, 'rider': 13, 'bus': 14, 'wall': 15})

LABEL_TASK['GTAV2Cityscapes'] = OrderedDict({'sky': 0, 'building': 1, 'road': 2, 'sidewalk': 3, 'fence': 4, 'vegetation': 5, 
             'pole': 6, 'car': 7, 'traffic sign': 8, 'pedestrian': 9, 'bicycle': 10, 'motorcycle': 11, 
             'traffic light': 12, 'rider': 13, 'bus': 14, 'wall': 15, 'terrain': 16, 'truck': 17, 'train': 18})


def get_label_map(source, target): 
    task = '%s2%s'%(source, target)
    assert(task in LABEL_TASK), task
    label_task = LABEL_TASK[task]
    ignore_label = cfg.DATASET.IGNORE_LABEL

    label_map = {source: {}, target: {}}
    for domain in [source, target]:
        assert(domain in LABELS), domain
        cur_label_map = LABELS[domain]
        for key in cur_label_map:
            ori_id = cur_label_map[key]
            if key in label_task:
                label_map[domain][ori_id] = label_task[key]
            else:
                label_map[domain][ori_id] = ignore_label

    return label_map
        

if __name__ == '__main__':
    label_map = get_label_map('SYNTHIA', 'Cityscapes')
    print(label_map)
    label_map = get_label_map('GTAV', 'Cityscapes')
    print(label_map)
