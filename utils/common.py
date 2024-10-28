def split_even_odd_indices(a):
    even_indices = []
    odd_indices = []
    
    for i in range(len(a)):
        if i % 2 == 0:
            even_indices.append(a[i])
        else:
            odd_indices.append(a[i])
    
    return even_indices, odd_indices

