import chardet

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

file_path = 'A:/YY/LivingWithBeautifulWomen.github.io/data/三国演义.txt'
file_path = 'A:/YY/LivingWithBeautifulWomen.github.io/data/西游记.txt'

encoding = detect_file_encoding(file_path)
print(f'The encoding of file {file_path} is {encoding}')

