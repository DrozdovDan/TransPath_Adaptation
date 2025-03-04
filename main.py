# %%
from dataset_generation.dataset_generator import generate_dataset_by_label
import matplotlib.pyplot as plt

# %%
if __name__ == '__main__':
    data = generate_dataset_by_label('bug_traps', 64)
    plt.imshow(data[43, :, :, 0], cmap='gray')
    plt.show()