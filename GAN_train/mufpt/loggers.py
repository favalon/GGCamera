class console_logger():
    def __call__(self, string):
        str_aux = string + '\n' if string[-1] != '\n' else string
        print(str_aux)


class file_logger():
    def __init__(self, file_path, append=True):
        self.file_path = file_path

        if not append:
            open(self.file_path, 'w').close()

    def __call__(self, string):
        str_aux = string + '\n'

        f = open(self.file_path, 'a')
        f.write(str_aux)
        f.close()
