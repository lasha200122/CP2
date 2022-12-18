import TextEncDec

key = [[1, 100, 10], [3, 2, 2], [12, 132, 1]]
Text = "Let's try our algorithm"
path = "picture.png"
encrypt_path = "encrypt.png"


if __name__ == '__main__':
    text_class = TextEncDec.Transform(Text, key, path, encrypt_path)
