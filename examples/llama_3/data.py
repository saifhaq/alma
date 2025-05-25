import random, string
from torch.utils.data import Dataset, Sampler
import itertools

BASE_PROMPTS = [
    "In a world where technology and nature have merged, cities now grow like forests and buildings bloom like flowers. The year is 2150, and I am walking through the streets of Neo-Singapore, where:",
    "The quantum computer finally solved the problem that had puzzled scientists for centuries. The implications were enormous:",
    "As the last human colony on Mars celebrated its centennial, an unexpected signal was received from deep space. The message contained:",
    "The AI poet had finally learned to feel emotions. Its first sonnet about love began:",
    "In the aftermath of the Great Climate Restoration, ecosystems thought extinct for decades began to re-emerge. The first signs appeared in:",
    "The kitchen of the future was not what anyone expected. Instead of robots, it featured:",
    "When time travel became commercially available, the most popular destination wasn't the past or future, but rather:",
    "The neural interface allowed people to share memories directly. The most surprising effect of this technology was:",
    "On the 500th anniversary of the first Mars landing, the president of the United Solar System gave a speech that started with:",
    "The library of forgotten books contained volumes that had been written but never published. Among them was a manuscript that:",
]


def _randomword(length):
    """
    Helper function to create a random word we add to base prompt to avoid cache hits on 
    prompts.

    Inputs:
    length (int): the length of the random words in chars.

    Outputs:
    (str): a string of `length` random underscore chars.
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

class PromptDataset(Dataset):
    def __init__(self, include_random_prefix: bool = False, prompts: list[str] = BASE_PROMPTS):
        self.prompts = prompts
        self.include_random_prefix: bool = include_random_prefix

        # We have the option to include a random string at the start of the prompt to stop any
        # cache hits
        if include_random_prefix:
            self.__getitem__ = self.__getitem_with_rand_prefix__
    
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

    def __getitem_with_rand_prefix__(self, idx):
        return f"{_randomword(10)} - {self.prompts[self.index]}"


# Custom sampler that cycles through the dataset indefinitely
class CircularSampler(Sampler):
    def __init__(self, data_source, total_samples: int):
        self.data_source = data_source
        self.total_samples = total_samples
    
    def __iter__(self):
        # Repeat indices to achieve the total number of samples needed
        return itertools.islice(
            itertools.cycle(range(len(self.data_source))),
            self.total_samples
        )
    
    def __len__(self):
        return self.total_samples
