import copy
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import random
from sklearn.metrics import confusion_matrix





def plot_diff_cluster(y, y_pred, data_df):
    """
        PLOT_DIFF_CLUSTER - Show the misclassification of two clusterings.

        INPUT:
            y: First (true) clustering 
            y_pred: Second (predicted) clustering
            data_df: Full data
        OUTPUT:
            VOID - Plot the diff figure.

        This function assumes that the number of clusters is identical in both clustering.
        The author is aware of the ugliness of his function, feel free to not contact him.

        Author: Loïc Van Hoorebeeck
        Date: 2020-10-15
    """
    
    # Indexing change in the case when the true classification is not sorted
    idx = np.argsort(y)
    y = y[idx]
    y_pred = y_pred[idx]
    data_df = data_df.reindex(idx)

    labels = list(np.unique(y))

    
    labels = [0, 1, 2]
    labels_pred = []
    _labels = list(labels)

    nums_labels = [list(map(lambda x: len(x), np.where(y == l)))[0] for l in labels]
    cums_nums_labels = np.cumsum(nums_labels)

    # Gets the predicted labels, this step is probably useless
    C = confusion_matrix(y, y_pred, labels=labels)
    _C = copy.copy(C)
    for c in range(C.shape[1]-1):
        _max = np.max(_C[_labels, c])
        i_max = int(np.where(_C[:, c] == _max)[0])
        _labels.remove(i_max)
        labels_pred.append(i_max)
    x = [i for i in labels if i not in labels_pred]
    labels_pred = labels_pred + x
    C = C[labels_pred, :]

    for l in labels[:-1]:
        # Select nums_labels[l] lines of y_pred corresponding to labels[l]
        # then find which probable label corresponds to l 
        
        nums_labels_l = [list(map(lambda x: len(x),
                                  np.where(y_pred[cums_nums_labels[l]-nums_labels[l]:cums_nums_labels[l]] == _l)))[0]
                         for _l in labels_pred]


        idx_max = np.argmax(nums_labels_l)
        l_prob = labels_pred[idx_max]  # probable label

        # Switch the labels
        i_l_prob = np.where(y_pred == l_prob)
        i_old_l = np.where(y_pred == l)
        y_pred[i_l_prob] = l
        y_pred[i_old_l] = l_prob

    # Error computation and plot
    err = [0 if s == t else 1 for (s, t) in zip(y_pred, y)]

    fig = px.scatter_3d(data_df, x=data_df.columns[0], y=data_df.columns[1], z=data_df.columns[2],
                  color=err)
    fig.update_layout(coloraxis_colorbar=dict(
        yanchor="top", y=1,
        title="Classification",
        tickvals=[0, 1],
        ticktext=["Correct", "Incorrect"],
        ticks="outside"
    ))
    fig.show()


def visualize_k_means(data, kmeans):
    """
        VISUALIZE_k_means - Visualize K-means clustering.

        INPUT:
            data - data used to build the model.
            kmeans - k-means clustering model.
        OUTPUT:
            VOID - Plot the clustering for the whole region (+/- 1) as well as the centroids.

        Largely inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
        Date: 2020-11-04
    """
    
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])


    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
                          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                          cmap=plt.cm.Paired,
                          aspect='auto', origin='lower')

    plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_


    plt.scatter(centroids[:, 0], centroids[:, 1],
                            marker='x', s=169, linewidths=3,
                            color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                        'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


