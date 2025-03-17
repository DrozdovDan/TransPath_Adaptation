import random
import numpy as np


def mixture_of_betas(alpha1, beta1, alpha2, beta2, w=0.5):
    # w – "вес" первой беты
    if np.random.random() < w:
        return np.random.beta(alpha1, beta1)
    else:
        return np.random.beta(alpha2, beta2)


def generate_QRs():
    size = 64
    QR = np.ones((size, size), dtype=np.float32)  # Все клетки изначально проходимые

    alpha1, beta1, alpha2, beta2 = (2, 2, 2, 2)
    k = mixture_of_betas(alpha1, beta1, alpha2, beta2)
    # Первый проход
    for i in range(0, size):
        for j in range(0, size):
            if random.random() < k:
                QR[i, j] = 0

    # Второй
    for _ in range(20):
        trap_x = random.randint(0, size - 1)
        trap_y = random.randint(0, size - 1)
        QR[trap_x, trap_y] = 0

    return QR
