from .detector_info import *

def build_object_detector(config):
    detector_impl = config['detector_impl']
    detector_name = config['detector_name']
    input_size = config['detector_input_size']
    test_size = config['detector_test_size']
    target_object_id = config['target_object_id']
    object_dataset = config['object_dataset']

    model_name = detector_name + '_' + object_dataset
    detector_info = DETECTOR_INFO[detector_impl][model_name]
    detector= detector_info['detector']
    cfg_path = detector_info['cfg_path']
    model_path = detector_info['model_path']

    if test_size[0] < 0:
        test_size = detector_info['test_size']

    if target_object_id < 0:
        target_object_id = detector_info['target_object_id']

    print ('====== Object Detector Information ========')
    print ('Detector: %s: %s...' % (detector_impl, detector_name))
    print ('CFG path: %s Model path: %s ' % (cfg_path, model_path))
    print ('Input_size: (%d %d) test_size (%d %d)' % (input_size[0], input_size[1], test_size[0], test_size[1]))
    print ('Dataset: %s' % (object_dataset))
    print ('Target object ID: %d\n' % (target_object_id))

    return detector(model_name=model_name,
                        cfg_path=cfg_path,
                        model_path= model_path,
                        class_names = OBJECT_CLASS_NAMES[object_dataset],
                        input_size=input_size,
                        test_size=test_size,
                        target_object_id = target_object_id)
