from config import CHEXPERT_CLASSES, CXR_IRGEN_CLASSES


def cxrigen_label_to_chexpert(cxrigen_label_vector):
    """Map a 14-dim CXR-IRGen label vector into CheXpert column ordering."""
    chexpert_labels = [0.0] * len(CHEXPERT_CLASSES)
    for irgen_idx, irgen_name in enumerate(CXR_IRGEN_CLASSES):
        for cxpert_idx, cxpert_name in enumerate(CHEXPERT_CLASSES):
            if irgen_name == cxpert_name.lower():
                chexpert_labels[cxpert_idx] = float(cxrigen_label_vector[irgen_idx])
                break
    return chexpert_labels


def get_intended_chexpert_index(cxrigen_label_vector):
    """Return the CheXpert index of the dominant class in a CXR-IRGen label vector."""
    import numpy as np
    intended_irgen_idx = int(np.argmax(cxrigen_label_vector))
    intended_name = CXR_IRGEN_CLASSES[intended_irgen_idx]
    for cxpert_idx, cxpert_name in enumerate(CHEXPERT_CLASSES):
        if cxpert_name.lower() == intended_name:
            return cxpert_idx, intended_name
    return None, intended_name
