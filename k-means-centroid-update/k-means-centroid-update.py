def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    centroids = [[[0 for _ in range(len(points[0]))]] for _ in range(k)]
    for point, assignment in zip(points, assignments):
        centroids[assignment].append(point)
    centroids = [[sum(p[i] for p in centroid) / (len(centroid) - 1 if len(centroid) > 1 else len(centroid))
                  for i in range(len(points[0]))] for centroid in centroids]
    return centroids