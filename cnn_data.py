from sys import stdout
import numpy
import scipy.misc

from tokenizer import Tokenizer

class CnnDataIter():
    """This is an iterator class for preparing cnn batch data.
    
    Attributes:
        data_path (str): The path where to access the data.
        images_path (str): The path where to access the images.
        formulas_file (str): The file records the formula list.
        dataset_file (str): The file records the images and corresponding formulas in the dataset.        
        
        box (dict): The region of image to be cropped.
        
        batch_size (int): This is the number of batch dataset size.
        
        labels ([dict]): This is the list of labels for the dataset, where the label is a dictionary.
        image_files ([str]): This is the list of image files for the dataset, where the entry is the file path.
        dataset_size (int): The size of the datasize.
        indices ([int]): The list of random unique indices to access the data.
        
        position (int): The current start position of the batch data.
    
    """
    def __init__(self, dataset, batch_size = 128):
        """
        
        Notes:
            `dataset` parameter must in ['train', 'test', 'validate']
            
        Args:
            dataset: This is the dataset where the batch dataset is from.
            batch_size (int): This is the number of batch dataset size.
        
        """
        self.data_path = 'data/'
        self.images_path = 'data/formula_images/'
        self.formulas_file = self.data_path + 'im2latex_formulas.lst'
        if dataset in ['train', 'test', 'validate']:
            self.dataset_file = self.data_path + 'im2latex_' + dataset + '.lst'
        else:
            print "Not legal dataset."
            raise
        
        self.box = {'top': 419, 'left': 558, 'bottom': 577, 'right': 1132}
        self.batch_size = batch_size        
        
        self.prepare()
        self.reset()
        
    def prepare(self):
        """This is an auxilary method for __init__.
        
        This method initialize the member labels and images    
        """
        def formula_tokenizer(formula):
            """This is a helper function to tokenize the formula to the label
            
            Args:
                formula (str): The Latex formula string.
                
            Returns:
                tokens_dict (dict): The dictionary recording how many numbers for each token in the formula.
                
            """
            tokens_dict = {}
            tokenizer = Tokenizer(formula)
            for token in tokenizer.tokenize():
                # formation of token: (line_number, type, token_part, token_full)        
                # command token process
                if token[1] == 'command':
                    if '\\'+token[2] not in tokens_dict:
                        tokens_dict['\\'+token[2]] = 1
                    else:
                        tokens_dict['\\'+token[2]] += 1
                # text token process
                elif token[1] == 'text':
                    for c in token[3]:
                        if c not in tokens_dict:
                            tokens_dict[c] = 1
                        else:
                            tokens_dict[c] += 1
            return tokens_dict
        
        print "preparing data...",
        
        formula_list = []
        labels = []
        image_files = []
        
        # retrieve the formulas list from the file
        with open(self.formulas_file) as formulas_file:
            for line in formulas_file:
                formula_list.append(line)
        
        # according to the dataset_file to initialize the images and labels 
        with open(self.dataset_file, 'r') as examples:
            for example in examples:
                label = formula_tokenizer(formula_list[int(example.split()[0])])
                image_file = example.split()[1] + '.png'
                labels.append(label)
                image_files.append(image_file)
        
        self.dataset_size = len(labels)
        self.labels = labels
        self.image_files = image_files
        self.indices = range(self.dataset_size)
        
        print "done."
    
    def reset(self):
        """The method to reset indices and position after the dataset finish a round."""
        self.indices = numpy.random.permutation(self.indices)
        self.position = 0
    
    def next(self):
        """The method to iterate over the dataset and return batch data.
        
        Returns:
            labels ([dict]): The batch labels.
            images (ndarray): The batch images with shape (batch_size, height, width).
        
        """
        # check if the dataset is finished
        if self.position >= self.dataset_size:
            self.reset()
            raise StopIteration()
        
        # get the batch size
        rest_size = self.dataset_size - self.position
        batch_size = numpy.minimum(self.batch_size, rest_size)
        # get the indices for the current batch
        current_indices = self.indices[self.position:self.position+batch_size]
        self.position += batch_size
        
        # get the labels and image_files corresponding to indices
        labels = [self.labels[idx] for idx in current_indices]
        image_files = [self.image_files[idx] for idx in current_indices]
        
        # crop the images
        images = numpy.empty((batch_size, self.box['bottom'] - self.box['top'], self.box['right'] - self.box['left']))
        for i in range(len(image_files)):
            image = scipy.misc.imread(self.images_path + image_files[i], flatten = True)
            image = image[self.box['top']:self.box['bottom'], self.box['left']:self.box['right']]
            image = 255 - image
            
            images[i,:,:] = image
            
        return (labels, images)
    
    def __iter__(self):
        return self