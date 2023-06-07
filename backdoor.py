import copy

def LSBsteganography(img,message):
    ascii_value = [ord(x) for x in message]
    bin_message = [bin(x) for x in ascii_value]
    bin_message = [b[2:].zfill(8) for b in bin_message]
    bin_num_message = []
    for i in bin_message:
        for j in i:
            bin_num_message.append(int(j)) 
    output = copy.copy(img)
    embed_counter = 0
    for c in range(3):
        for i in range(img.shape[0]): 
            for j in range(img.shape[1]):
                if (embed_counter < len(message) * 8):
                    LSB = float(img[i,j,c])% 2
                    temp = float( bool(LSB) != bool(bin_num_message[embed_counter]))
                    output[i,j,c] = img[i,j,c]+temp
                    embed_counter += 1
            
                elif embed_counter % len(message) * 8 == 0:
                    embed_counter = 0
    
    return output
    

