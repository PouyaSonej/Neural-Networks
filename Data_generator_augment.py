class DataGeneratorAugment(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, batch_size=32, dim=(224, 224), n_channels=3, n_classes=196, shuffle=True, mode='test'):
        'Initialization'
        self.data = data
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mode = mode
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ind in enumerate(indexes):
            line = self.data[ind][0].split(',')
            class_id = int(line[5]) - 1
            image = cv2.imread('/content/' + line[0])
            image = cv2.resize(image, self.dim)

            if self.mode == 'train':
                image = seq(image=image)

            image = preprocess_input(image)

            # Store sample
            X[i,] = image

            # Store class
            y[i] = class_id

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
            
   ###############################################################3
train_generator = DataGeneratorAugment(train_list, mode='train')
valid_generator = DataGeneratorAugment(test_list, shuffle=False)
