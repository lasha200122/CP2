import Helper


class Transform:
    def __init__(self, text, K, path, encrypt_path):
        self.binTexts = []
        if not Helper.gram_schmidt(K):
            print("The current Key is not invertible choose another matrix")
        else:
            values = Helper.Encrypt(K, text)
            for value in values:
                self.binTexts.append(Helper.number_to_bin(value))
            Helper.hideInTheImage(path, self.binTexts, encrypt_path)
            bin_letters = Helper.GetNumbersFromImage(encrypt_path)
            characters = Helper.getLetterOrders(bin_letters, K)
            result = ''.join([chr(i) for i in characters])
            print(result)
