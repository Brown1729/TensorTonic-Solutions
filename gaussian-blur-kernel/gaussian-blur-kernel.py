def gaussian_kernel(size, sigma):
    x = [i - (size // 2) for i in range(size)]
    dec = 2 * sigma ** 2
    x = [math.exp(- (i ** 2) / dec) for i in x]
    kernel = [[i * j for i in x] for j in x]
    sum_ = sum(sum(i) for i in kernel)
    kernel = [[i / sum_ for i in j] for j in kernel]
    return kernel