import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn

def fit_gmm(data, K=5, verbose=False):
    def likelihood(means, covariances, mixing_proportions):
        # Compute data log-likelihood for the given GMM parametrization
        densities = np.array([mvn.pdf(data, means[k], covariances[k]) for k in range(K)])
        unnormalized_responsibilities = densities * mixing_proportions
        return np.log(unnormalized_responsibilities.sum(axis=0)).sum()

    data = data.reshape(-1, data.shape[-1])

    N = data.shape[0]
    D = data.shape[1] # Dimension of the data points
    
    # Initialize the variables that are to be learned
    covariances = np.array([100 * np.eye(D) for k in range(K)]) # Covariance matrices
    mixing_proportions = np.ones([K, 1]) / K # Mixing propotions
    responsibilities = np.zeros([N, K])
    
    # Choose the initial centroids using k-means clustering
    kmeans = KMeans(n_clusters=K)
    kmeans = kmeans.fit(data)
    means = kmeans.cluster_centers_
    
    old_likelihood = likelihood(means, covariances, mixing_proportions)
    
    if verbose:
        print("Likelihood after intialization: {0:.2f}".format(old_likelihood))
        
    # Iterate until convergence
    it = 0
    converged = False
    while not converged:
        it += 1
        old_likelihood = likelihood(means, covariances, mixing_proportions)

        # Compute the responsibilities
        densities = np.array([mvn.pdf(data, means[k], covariances[k]) for k in range(K)])
        responsibilities = densities * mixing_proportions
        responsibilities = (responsibilities / responsibilities.sum(axis=0)).T

        # Update the distribution parameters
        resp_sums = responsibilities.sum(axis=0)
        means = responsibilities.T.dot(data)
        for k in range(K):
            means[k] /= resp_sums[k]
            covariances[k] = np.zeros(D)
            for n in range(N):
                centered = data[n, :] - means[k]
                covariances[k] += responsibilities[n, k] * np.outer(centered, centered)
            covariances[k] /= resp_sums[k]
            covariances[k] += 0.1 * np.eye(D) # To prevent singular matrices
        mixing_proportions = np.reshape(resp_sums / N, [K, 1])

        # Check for convergence
        new_likelihood = likelihood(means, covariances, mixing_proportions)
        delta = new_likelihood - old_likelihood
        converged = delta < np.abs(new_likelihood) * 1e-4
        if verbose:
            print("Iteration {0}, likelihood = {1:.2f}, delta = {2:.2f}".format(it, new_likelihood, delta))
            
    return (means, covariances, mixing_proportions)


def show_segmented(img, means, covariances, mixing_proportions, threshold=1e-7, title=''):
    K = means.shape[0]
    flat_img = img.reshape([-1, img.shape[-1]])
    probas = np.array([mixing_proportions[k] * mvn.pdf(flat_img, means[k], covariances[k]) for k in range(K)]).T
    probas = probas.sum(axis=1)
    plt.figure(figsize=[10, 10])
    plt.imshow((probas > threshold).reshape(img.shape[:2]))
    plt.title(title)


def segmentation_with_gmm(box, n_components=5, img_name='banana.png'):
    img = cv2.imread(img_name)
    x1, x2, y1, y2 = box

    # Fitting on data within the box
    in_box = img[y1: y2, x1: x2, :]
    means, covariances, mixing_proportions = fit_gmm(in_box, K=n_components)
    show_segmented(img, means, covariances, mixing_proportions,
                   title='Training on the data inside the box')
    
    # Fitting on data outside the box
    xrange = set(range(x1, x2))
    yrange = set(range(y1, y2))
    outside_box = []
    
    for (x, y), val in np.ndenumerate(img[:, :, 0]):
        if x not in xrange and y not in yrange:
            outside_box.append(img[x, y, :])

    outside_box = np.array(outside_box)
    means, covariances, mixing_proportions = fit_gmm(outside_box, K=n_components)
    show_segmented(img, means, covariances, mixing_proportions, threshold=1e-6,
                   title='Training on the data outside the box')

    plt.show()

if __name__ == '__main__':
    box = [200, 520, 280, 370]
    segmentation_with_gmm(box)