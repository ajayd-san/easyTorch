from skimage import io, color


class DatasetCleaner:
    def __init__(self, paths):
        self.invalid_paths = []
        self.paths = paths.iloc[:, 0]
        self.dataset_length = paths.shape[0]

    def fit(self):
        for idx in range(self.dataset_length):
            try:
                image = io.imread(self.paths[idx])
                if image.ndim == 2:
                    image = color.gray2rgb(image)
                if image.shape[2] > 3:
                    image = color.rgba2rgb(image)
            except:
                self.invalid_paths.append(idx)

    def transform(self):
        return self.invalid_paths

    def fit_transform(self):
        self.fit()
        return self.invalid_paths
