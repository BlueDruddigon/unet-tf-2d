import glob
import os

import numpy as np
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for synapse dataset."""
    
    VERSION = tfds.core.Version('0.1.1')
    RELEASE_NOTES = {
      '0.1.1': 'Initial release.',
    }
    
    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
          features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(None, None, 1), dtype=np.float32),
            'label': tfds.features.Image(shape=(None, None, 1), dtype=np.uint8, use_colormap=True),
          }),
          supervised_keys=('image', 'label'),
        )
    
    def _split_generators(self, _: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO: change this
        root_dir = ''
        
        return {
          'train': self._generate_examples(os.path.join(root_dir, 'train')),
          'valid': self._generate_examples(os.path.join(root_dir, 'valid')),
          'test': self._generate_examples(os.path.join(root_dir, 'test')),
        }
    
    def _generate_examples(self, path):
        """Yields examples."""
        for idx, f in enumerate(glob.glob(f'{path}/*.npz')):
            data = np.load(f)
            image, label = data['image'], data['label']
            data.close()
            
            yield idx, {
              'image': np.expand_dims(image, axis=-1).astype(np.float32),
              'label': np.expand_dims(label, axis=-1).astype(np.uint8),
            }
