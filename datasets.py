import torch
from torch.utils.data import Dataset, Subset
import os
import h5py
import numpy as np
from PIL import Image
from collections import defaultdict

# ================================
# GENERIC BALANCING FUNCTION
# ================================
def balance_split(images, concepts, labels, max_per_class=None, seed=42):
    """
    Given arrays of images, concepts, and labels, return balanced arrays with equal class counts.
    Args:
        images: np.ndarray, shape (N, ...)
        concepts: np.ndarray, shape (N, ...)
        labels: np.ndarray, shape (N,)
        max_per_class: int or None, maximum samples per class (default: min class count)
        seed: int, random seed
    Returns:
        balanced_images, balanced_concepts, balanced_labels
    """
    
    np.random.seed(seed)
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[int(label)].append(idx)
    min_class_count = min(len(idxs) for idxs in class_indices.values())
    if max_per_class is not None:
        min_class_count = min(min_class_count, max_per_class)
    selected_indices = []
    for idxs in class_indices.values():
        idxs = np.array(idxs)
        if len(idxs) > min_class_count:
            idxs = np.random.choice(idxs, min_class_count, replace=False)
        selected_indices.extend(idxs)
    np.random.shuffle(selected_indices)
    balanced_images = images[selected_indices]
    balanced_concepts = concepts[selected_indices]
    balanced_labels = labels[selected_indices]
    return balanced_images, balanced_concepts, balanced_labels

# ================================
# FUNCTIONS FOR SHAPES3D DATASET 
# ================================

def one_hot_concepts(concepts):
    I = np.unique(concepts)
    one_hots = []
    diag_matrix = np.eye(len(I))
    for sample in range(len(concepts)):
        for i in range(len(I)):
            if concepts[sample] == I[i]:
                one_hots.append(diag_matrix[i])
    one_hots = np.stack(one_hots)
    return one_hots

def shapes_3d_base(base_path='shapes3d'):
    print('Loading the dataset...')
    print(base_path)
    path= os.path.join(base_path, '3dshapes.h5')
    with h5py.File(path, 'r') as f:

        images = f['images']
        concepts = f['labels']

        images   = images[()]
        concepts = concepts[()]
        labels   = np.copy(concepts)
    
    # Description of the dataset concepts (3D shapes)
    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}
    '''     
            0.0 = red
            0.1 = orange
            0.2 = yellow
            0.7 = blue
    '''

    ## CREATE LABELS FOR CLASSIFYING THE PILL SHAPE, RED, any size, any orientation
    predictions = []
    for j in range(labels.shape[0]):
        if labels[j,4] == 3 and labels[j,2] == 0.0:
            predictions.append(1)
        else:
            predictions.append(0)

    labels = np.array(predictions)
    print(labels.shape)

    object_hue = concepts[:, 2]  # object_hue
    shape = concepts[:, 4]       # shape

    object_hue_onehot = one_hot_concepts(object_hue)  # (N, 10)
    shape_onehot = one_hot_concepts(shape)            # (N, 4)

    preprocess_concepts = np.hstack([object_hue_onehot, shape_onehot])

    print(preprocess_concepts.shape)
    for i in range(5):
        print(np.unique(preprocess_concepts[:,i]), i)
   
    return images, preprocess_concepts, labels

def create_dataset(base_path = 'shapes3d', output_dir=None):
    print('Creating the dataset to be saved...')
    # Load the dataset
    images, concepts, labels = shapes_3d_base(base_path)

    # Find all red pill and not red pill indices
    red_pill_indices = np.where(labels == 1)[0]
    not_red_pill_indices = np.where(labels == 0)[0]

    max_per_class = 12000
    np.random.shuffle(red_pill_indices)
    np.random.shuffle(not_red_pill_indices)
    red_pill_indices = red_pill_indices[:max_per_class]
    not_red_pill_indices = not_red_pill_indices[:max_per_class]

    n_train = int(0.6 * max_per_class)
    n_val = int(0.2 * max_per_class)
    n_test = max_per_class - n_train - n_val 

    train_red = red_pill_indices[:n_train]
    val_red = red_pill_indices[n_train:n_train+n_val]
    test_red = red_pill_indices[n_train+n_val:]

    train_notred = not_red_pill_indices[:n_train]
    val_notred = not_red_pill_indices[n_train:n_train+n_val]
    test_notred = not_red_pill_indices[n_train+n_val:]

    # Concatenate and shuffle
    train_idx = np.concatenate([train_red, train_notred])
    val_idx = np.concatenate([val_red, val_notred])
    test_idx = np.concatenate([test_red, test_notred])
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    # Subset of each split 
    max_train = None 
    max_val = None   
    max_test = None  
    if max_train is not None and len(train_idx) > max_train:
        train_idx = train_idx[:max_train]
    if max_val is not None and len(val_idx) > max_val:
        val_idx = val_idx[:max_val]
    if max_test is not None and len(test_idx) > max_test:
        test_idx = test_idx[:max_test]

    # Create splits
    train_images = images[train_idx]
    train_concepts = concepts[train_idx]
    train_labels = labels[train_idx]
    train_images, train_concepts, train_labels = balance_split(train_images, train_concepts, train_labels)

    val_images = images[val_idx]
    val_concepts = concepts[val_idx]
    val_labels = labels[val_idx]
    val_images, val_concepts, val_labels = balance_split(val_images, val_concepts, val_labels)

    test_images = images[test_idx]
    test_concepts = concepts[test_idx]
    test_labels = labels[test_idx]
    test_images, test_concepts, test_labels = balance_split(test_images, test_concepts, test_labels)

    np.save(os.path.join(output_dir, 'train_split_imgs_clean.npy'), train_images)
    np.save(os.path.join(output_dir, 'train_split_cl_clean.npy'), np.hstack((train_concepts, train_labels.reshape(-1,1))))
    np.save(os.path.join(output_dir, 'val_split_imgs_clean.npy'), val_images)
    np.save(os.path.join(output_dir, 'val_split_cl_clean.npy'), np.hstack((val_concepts, val_labels.reshape(-1,1))))
    np.save(os.path.join(output_dir, 'test_split_imgs_clean.npy'), test_images)
    np.save(os.path.join(output_dir, 'test_split_cl_clean.npy'), np.hstack((test_concepts, test_labels.reshape(-1,1))))

    # Inject noise into concepts
    noise_prob = 0.2
    mask = np.random.rand(*train_concepts.shape) < noise_prob
    train_concepts[mask] = 1 - train_concepts[mask]

    # Inject noise into labels
    label_noise_prob = 0.1
    label_mask = np.random.rand(*train_labels.shape) < label_noise_prob
    train_labels[label_mask] = 1 - train_labels[label_mask]

    print('Saving dataset')
    np.save(os.path.join(output_dir, 'train_split_imgs.npy'), train_images)
    np.save(os.path.join(output_dir, 'train_split_cl.npy'), np.hstack((train_concepts, train_labels.reshape(-1,1))))
    np.save(os.path.join(output_dir, 'val_split_imgs.npy'), val_images)
    np.save(os.path.join(output_dir, 'val_split_cl.npy'), np.hstack((val_concepts, val_labels.reshape(-1,1))))
    np.save(os.path.join(output_dir, 'test_split_imgs.npy'), test_images)
    np.save(os.path.join(output_dir, 'test_split_cl.npy'), np.hstack((test_concepts, test_labels.reshape(-1,1))))

    print('Dataset saved!')
    print('Train:', len(train_images))
    print('Val:', len(val_images))
    print('Test:', len(test_images))


# ================================
# SHAPES3D DATASET 
# ================================
SHAPES3D_CONCEPT_NAMES = [
    'red', 'orange', 'yellow', 'lime', 'green', 'sky blue', 'blue', 'dark blue', 'purple', 'pink',
    'cube', 'cylinder', 'sphere', 'pill'
]

class SHAPES3DOriginal(Dataset):
    # name = "shapes3d_original"
    def __init__(self, root='shapes3d', split='train', transform = None, args=None, output_dir='pipes_shapes3d'):
        self.base_path = os.path.join(root, '3dshapes.h5')
        self.output_dir = output_dir

        if not os.path.exists(os.path.join(output_dir, split+'_split_imgs.npy')) or not os.path.exists(os.path.join(output_dir, split+'_split_cl.npy')):
            create_dataset(root, output_dir=output_dir)
        
#        if split == 'train':
#            self.images= np.load(os.path.join(output_dir, split+'_split_imgs.npy'), allow_pickle=True)
#            concepts_and_labels = np.load(os.path.join(output_dir, split+'_split_cl.npy'), allow_pickle=True)
#        else:  
        self.images = np.load(os.path.join(output_dir, split+'_split_imgs_clean.npy'), allow_pickle=True)
        concepts_and_labels = np.load(os.path.join(output_dir, split+'_split_cl_clean.npy'), allow_pickle=True)
        
        self.classes = ['not red pill', 'red pill']
        self.labels = concepts_and_labels[:,-1]
        self.concepts = concepts_and_labels[:,:-1]
        self.transform = transform

        
    def __getitem__(self, idx):
        image = self.images[idx]
        # Convert the image to PIL img
        image = Image.fromarray(image)
        concepts = self.concepts[idx]
        labels = self.labels[idx]
        if self.transform is not None:
            return self.transform(image), torch.tensor(concepts), torch.tensor(labels)
        else: 
            return image, concepts, int(labels)

    def __len__(self):
        return len(self.images) 

    def getoutput(self):
        return self.output_dir

class SHAPES3D_Custom(Subset):
    def __init__(self, root='shapes3d', split='train', transform=None, args=None, output_dir=None):
        original_dataset = SHAPES3DOriginal(root=root, split=split, transform=transform, args=args, output_dir=output_dir)
        indexes = range(len(original_dataset))
        output_dir = original_dataset.getoutput()

        super().__init__(original_dataset, indexes)

        self.output_dir = output_dir
        self.classes = original_dataset.classes
        self.images = original_dataset.images
        self.concepts = original_dataset.concepts
        self.labels = original_dataset.labels

class SHAPES3DMini(Subset):
    name = "shapes3d_mini"
    def __init__(self, root='shapes3d', split='train', transform = None, args=None, subset_indices = [0,1000], output_dir='pipes_shapes3d'):
        self.data = SHAPES3D_Custom(root=root, split=split, transform=transform, args=args, output_dir=output_dir)
        self.classes = self.data.classes
        self.transform = transform
        start, end = subset_indices[0], subset_indices[1]
        n = len(self.data)
        if end > n:
            print(f"[SHAPES3DMini] Requested end index {end} exceeds dataset size {n}. Using end={n}.")
            end = n
        if start < 0:
            print(f"[SHAPES3DMini] Requested start index {start} is less than 0. Using start=0.")
            start = 0
        if start >= end:
            print(f"[SHAPES3DMini] Invalid indices: start={start}, end={end}, dataset size={n}. Using full available range.")
            start = 0
            end = n
        self.output_dir = output_dir
        self.subset_indices = list(range(start, end))
        super().__init__(self.data, self.subset_indices)

# ================================
# CUB-200-2011 DATASET OVERVIEW
# ================================

CUB_CONCEPT_NAMES = [
    'forehead_black', 'crown_black', 'nape_black', 'throat_black', 
    'breast_black', 'back_black', 'upperparts_black', 'belly_black', 
    'wing_black', 'upper_tail_black', 'under_tail_black', 'primary_black' 
]

class CUBDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', val_ratio=0.2, random_seed=42):
        """
        CUB-200-2011 Dataset with concept annotations
        Modified for "Black Bird vs Not Black Bird" binary classification
        Args:
            data_dir (str): Path to CUB dataset directory
            transform: Image transformations
            split (str): 'train', 'val', 'test'
            val_ratio (float): Fraction of training data to use for validation
            random_seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        # Define black-related concept indices
        self.black_concept_indices = self.extract_black_concept_indices(data_dir)
        self.classes = ['not black bird', 'black bird']
        
        # Load dataset files
        self._load_data()

    def load_attribute_names(self, data_path):
        """Load attribute names from attributes.txt"""
        attributes_txt = os.path.join(data_path, 'attributes', 'attributes.txt')
        attribute_names = {}
        
        try:
            with open(attributes_txt, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        attr_id = int(parts[0])
                        attr_name = parts[1]
                        attribute_names[attr_id] = attr_name
            print(f"Loaded {len(attribute_names)} attribute names from {attributes_txt}")
        except FileNotFoundError:
            print(f"Warning: {attributes_txt} not found. Using default black concept indices.")
            return {}
            
        return attribute_names
        
    def extract_black_concept_indices(self, data_dir):
        """Extract indices of black-related concepts from CUB attributes"""
        black_concepts = {}
        attribute_names = self.load_attribute_names(data_dir) 
        for attr_id, attr_name in attribute_names.items():
            attr_lower = attr_name.lower()
            if 'black' in attr_lower:
                # Map common black color attributes to body parts
                if 'forehead' in attr_lower:
                    black_concepts['forehead_black'] = attr_id - 1  # Convert to 0-indexed
                elif 'crown' in attr_lower:
                    black_concepts['crown_black'] = attr_id - 1
                elif 'nape' in attr_lower:
                    black_concepts['nape_black'] = attr_id - 1
                elif 'throat' in attr_lower:
                    black_concepts['throat_black'] = attr_id - 1
                elif 'breast' in attr_lower:
                    black_concepts['breast_black'] = attr_id - 1
                elif 'back' in attr_lower:
                    black_concepts['back_black'] = attr_id - 1
                elif 'upperparts' in attr_lower:
                    black_concepts['upperparts_black'] = attr_id - 1
                elif 'belly' in attr_lower:
                    black_concepts['belly_black'] = attr_id - 1
                elif 'wing' in attr_lower:
                    black_concepts['wing_black'] = attr_id - 1
                elif 'upper_tail' in attr_lower:
                    black_concepts['upper_tail_black'] = attr_id - 1
                elif 'under_tail' in attr_lower:
                    black_concepts['under_tail_black'] = attr_id - 1
                elif 'primary' in attr_lower:
                    black_concepts['primary_black'] = attr_id - 1
        return black_concepts

    def _load_txt_file(self, filepath):
        """Load a space-separated text file"""
        data = []
        try:
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        data.append(parts)
            print(f"Loaded {len(data)} entries from {filepath}")
        except FileNotFoundError:
            print(f"Warning: {filepath} not found.")
        return data

    def _create_splits(self, train_indices):
        """
        Create train/val split from the original training data
        Args:
            train_indices: Indices of samples marked as training in CUB split
        Returns:
            train_idx, val_idx: Indices for training and validation
        """
        np.random.seed(self.random_seed)
        
        # Shuffle
        shuffled_indices = np.random.permutation(train_indices)
        
        # Calculate split
        n_val = int(len(shuffled_indices) * self.val_ratio)
        n_train = len(shuffled_indices) - n_val
        
        train_idx = shuffled_indices[:n_train]
        val_idx = shuffled_indices[n_train:]
        
        print(f"Created splits from {len(train_indices)} training samples:")
        print(f"  New train: {len(train_idx)} samples")
        print(f"  New val: {len(val_idx)} samples")
        
        return train_idx, val_idx

    def _load_data(self):
        """Load all CUB dataset files and process for black bird classification"""
        # Load image paths and labels
        images_file = os.path.join(self.data_dir, 'images.txt')
        labels_file = os.path.join(self.data_dir, 'image_class_labels.txt')
        split_file = os.path.join(self.data_dir, 'train_test_split.txt')
        attributes_file = os.path.join(self.data_dir, 'attributes', 'image_attribute_labels.txt')
        
        image_data = self._load_txt_file(images_file)      # [(image_id, image_path), ...]
        label_data = self._load_txt_file(labels_file)      # [(image_id, class_id), ...]
        split_data = self._load_txt_file(split_file)        # [(image_id, is_training), ...]
            
        with open(attributes_file, 'r') as f:
            attr_data = [line.strip().split() for line in f.readlines()]
        
        # Filter by split
        self.image_paths = []
        self.original_labels = []  # Keep original 200 bird species labels
        self.all_concepts = []
        
        # Group attributes by image
        attr_dict = {}
        for attr_line in attr_data:
            img_id, attr_id, is_present = attr_line[:3]
            img_id = int(img_id)
            if img_id not in attr_dict:
                attr_dict[img_id] = [0] * 312  # 312 attributes in CUB
            attr_dict[img_id][int(attr_id) - 1] = int(is_present)
        
        all_train_indices = []
        all_test_indices = []
        all_image_paths = []
        all_original_labels = []
        all_concepts = []
        
        for i, (img_data, label_data_i, split_data_i) in enumerate(zip(image_data, label_data, split_data)):
            img_id = int(img_data[0])
            is_train = int(split_data_i[1])
            
            image_path = os.path.join(self.data_dir, 'images', img_data[1])
            original_label = int(label_data_i[1]) - 1  # Convert to 0-indexed
            concepts = attr_dict.get(img_id, [0] * 312)
            
            all_image_paths.append(image_path)
            all_original_labels.append(original_label)
            all_concepts.append(concepts)
            
            if is_train == 1:
                all_train_indices.append(i)
            else:
                all_test_indices.append(i)
        
        # Create train/val split from training data
        train_idx, val_idx = self._create_splits(all_train_indices)
        
        if self.split == 'train':
            selected_indices = train_idx
        elif self.split == 'val':
            selected_indices = val_idx
        elif self.split == 'test':
            selected_indices = all_test_indices
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")
        
        # Filter data based on selected indices
        self.image_paths = [all_image_paths[i] for i in selected_indices]
        self.original_labels = np.array([all_original_labels[i] for i in selected_indices])
        self.all_concepts = np.array([all_concepts[i] for i in selected_indices], dtype=np.float32)
        
        self._process_black_bird_classification()
        
    def _process_black_bird_classification(self):
        """
        Create binary classification: Black Bird vs Not Black Bird
        Extract only black-related concepts for the concept bottleneck
        """
        print("Processing CUB dataset for 'Black Bird vs Not Black Bird' classification...")
        
        # Extract only black-related concepts
        black_concept_list = list(self.black_concept_indices.values())
        self.black_concepts = self.all_concepts[:, black_concept_list]

        # # Create binary labels based on black concepts
        binary_labels = []
        black_bird_count = 0

        primary_idx = None
        if 'primary_black' in list(self.black_concept_indices.keys()):
            primary_idx = list(self.black_concept_indices.keys()).index('primary_black')

        weights = np.ones(len(black_concept_list))
        if primary_idx is not None:
            weights[primary_idx] = 5

        weighted_scores = np.dot(self.black_concepts, weights)

        weighted_threshold = max(3, np.mean(weights))
        for i in range(len(self.all_concepts)):
            is_primary = False
            if primary_idx is not None and self.black_concepts[i, primary_idx] == 1:
                is_primary = True
            if is_primary:
                binary_labels.append(1)
                black_bird_count += 1
            elif weighted_scores[i] >= weighted_threshold:
                binary_labels.append(1)
                black_bird_count += 1
            else:
                binary_labels.append(0)
        self.binary_labels = np.array(binary_labels)

        total_samples = len(self.binary_labels)
        black_percentage = 100 * black_bird_count / total_samples
        
        print(f"\nDataset processed ({self.split}):")
        print(f"  Total samples: {total_samples}")
        print(f"  Black birds: {black_bird_count} ({black_percentage:.1f}%)")
        print(f"  Not black birds: {total_samples - black_bird_count} ({100 - black_percentage:.1f}%)")
        print(f"  Using {len(self.black_concept_indices)} black-related concepts out of 312 total")
        print(f"  Black concepts: {list(self.black_concept_indices.keys())}")

    def get_concept_names(self):
        """Return the names of the black-related concepts used"""
        return list(self.black_concept_indices.keys())
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Use black-related concepts only
        concepts = self.black_concepts[idx]
        
        # Use binary label (black bird vs not black bird)
        label = self.binary_labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.FloatTensor(concepts), torch.LongTensor([label]).squeeze()
