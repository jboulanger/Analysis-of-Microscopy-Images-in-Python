import numpy as np

def manders_coefficients(mask1,mask2,im1=1,im2=1):
    """Compute Manders overlap coefficients"""
    intersect = np.logical_and(mask1, mask2)
    n1 = np.sum(mask1 * im1, dtype=float)
    n2 = np.sum(mask2 * im2, dtype=float)
    m1 = np.sum(intersect * im1, dtype=float) / n1 if n1 > 0 else 0
    m2 = np.sum(intersect * im2, dtype=float) / n2 if n2 > 0 else 0
    return ( m1, m2 )

def manders_coefficients(mask1,mask2,im1,im2):
    """Compute Manders overlap coefficients"""
    intersect = np.logical_and(mask1, mask2)
    n1 = np.sum(mask1 * im1, dtype=float)
    n2 = np.sum(mask2 * im2, dtype=float)
    m1 = np.sum(intersect * im1, dtype=float) / n1 if n1 > 0 else 0
    m2 = np.sum(intersect * im2, dtype=float) / n2 if n2 > 0 else 0
    return ( m1, m2 )

def colocalization(images, masks):
    from scipy.stats import pearsonr, spearmanr
    result = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            union = np.logical_and(masks[i], masks[j])
            pcc = pearsonr(images[i][union], images[j][union])[0]
            scc = spearmanr(images[i][union], images[j][union])[0]
            mcc = manders_coefficients(masks[i], masks[j], images[i], images[j])
            result.append({
                'index 1': i,
                'index 2': j,
                'Pearson correlation coefficient' : pcc,
                'Spearman correlation coefficient' : scc,
                'Manders correlation coefficient 1:2': mcc[0],
                'Manders correlation coefficient 2:1': mcc[1],
                })
    return result
